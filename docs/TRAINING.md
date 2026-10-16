# Qwen3.8-27B pretrained-weight adaptation on H100

**Chosen route: transfer the official pretrained Qwen3.8-27B weights, not a fresh 27B pretraining run.** Preserve all 64 layers (including 48 GDN blocks), original Q, gates, O, FFN and LM head. First train only a shared bridge from the output H32. Full CED prefill skipping/CSA2 sparse kernels, KL distillation, Q/O LoRA, and trained B0T remain **not implemented**.

## Hardware assumptions and truthful status

27.8B BF16 weights alone are approximately 51.8 GiB. The frozen main model still needs backward **activations** for gradients to flow into the new bridge through later layers; one H100 80GB is not guaranteed to fit. Start with seq=128, batch=1, no unqualified gradient checkpointing, and measure actual peak VRAM before expansion. No actual 27B GPU training has been verified by editing this repository.

## Stage 0 — pretrained teacher + B0 baseline (implemented commands; H100 pending)

Pin the Qwen HF SHA and exact Transformers, tokenizer and dataset revisions. The unmodified pretrained Qwen (eval/no_grad) supplies H32 and eight original post-KNorm, pre-RoPE K/V targets. Native original layer 36 (one-based) supplies the bridge's starting K/V and KNorm weights. Teacher feature shards are written to local \`artifacts/\` and should never be committed as model weights.

## Stage 1 — offline feature warmup (implemented)

From original layer 36, copy K/V projection weights and KNorm into the new bridge (default \`--init source36\`). For the diagnostic \`--init random_linear\`, initialize **only K/V projection matrices** randomly; KNorm remains copied from layer 36. The optimizer learns a common bridge representation matched against original per-layer K/V targets using normalized MSE. It is not a guarantee of lower next-token loss. Current defaults: AdamW, lr=2e-4, weight decay=0, 100 iterations, 64 feature positions, seed 17. Original 27B weights need not reside on the GPU during this offline warmup.

## Stage 2 — frozen-base causal CE adaptation (implemented entrypoint; H100 pending)

Reload **the same pinned pretrained** Qwen checkpoint, freeze all backbone weights (\`model.requires_grad_(False)\`), install one bridge, load the Stage-1 checkpoint, optimize bridge with token-level next-token CE. Defaults for first H100 gate: 16 optimizer steps, seq=128, microbatch=1, accumulation=1, AdamW lr=2e-5, weight decay=.01, grad clip 1.0. Record nonzero finite bridge gradients, optimizer updates, actual tokens seen, peak memory, output checkpoint and held-out CE. The first 16 steps are a smoke test, **not** 100k-token adaptation.

~~~bash
python scripts/extract_teacher_features.py --revision "$MODEL_REVISION" --data "$TRAIN_JSONL" \
  --seq-len 128 --max-examples 16 --out artifacts/teacher_features
python scripts/train_feature_projector.py --features artifacts/teacher_features \
  --init source36 --out artifacts/warmup --steps 100 --batch-positions 64
python scripts/train_27b.py --revision "$MODEL_REVISION" \
  --train-data "$TRAIN_JSONL" --val-data "$VAL_JSONL" \
  --projector-init source36 --warmup-checkpoint artifacts/warmup/adapter.pt \
  --seq-len 128 --steps 16 --lr 2e-5 --out artifacts/train_b1
~~~

The first commands require pinned model revision and distinct JSONL train/val prepared with [H100_COMMANDS.md](H100_COMMANDS.md). Training checkpoint metadata now records initialization policy and stage where available; old checkpoints missing provenance are **legacy/unknown**.

## Stage 3 — optional logits teacher KL + late Q/O LoRA (NOT implemented)

Only after Stage 2 quality checks, consider a separate experiment with teacher distillation and/or Q/O LoRA. A conceptual prospective loss is

$$
\mathcal L_{\mathrm{future}} =
\mathcal L_{\mathrm{CE}}+\lambda T^2D_{\mathrm{KL}}(p_T^T\Vert p_S^T).
$$

Do **not** claim the current \`train_27b.py\` computes KL or updates LoRA: it computes CE only and trains bridge parameters.

## Controls, research goals and failure handling

B0 is original untrained pretrained Qwen for a fixed-quality anchor; **B0T matched training is not yet implemented**, but mandatory to identify architecture-specific effects. Compare \`source36\` versus \`random_linear\` initializations and feature warmup versus no warmup with exactly recorded datasets, random seeds, token exposures and full feature-extraction + GPU costs. Report unique KV bytes separately from full VRAM, GDN state size, TTFT and decode speed. This B1b path preserves all 64 blocks and does not reduce decoder GDN prefill FLOPs. Keep logs/checkpoints on the actual host; use GitHub only for reviewed source/evidence direct pull/push (no Actions).

See [WEIGHTS_AND_INITIALIZATION.md](WEIGHTS_AND_INITIALIZATION.md), [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) and [QWEN38_NATIVE_COMPATIBILITY.md](QWEN38_NATIVE_COMPATIBILITY.md).
