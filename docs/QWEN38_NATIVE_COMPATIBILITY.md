# Qwen3.8-27B native compatibility audit — B1b only

**Status:** design and source-contract audit completed; tiny/mock PyTorch tests passed previously. **Actual Qwen3.8-27B on H100 remains UNVERIFIED.** This is an adapter to Qwen's hybrid architecture, not a faithful recreation of DeepSeek-V4.1 CED+CSA2.

## 1. What does not need to be removed?

| Component | Qwen original | B1b change | Can it be deleted? |
|---|---|---|---|
| 64 decoder blocks | 48 GDN + 16 GQA | All 64 still execute | **No** |
| First 32 blocks | 24 GDN + 8 GQA | Produce boundary output H32 | **No** |
| Later 24 GDN blocks | Per-layer recurrent + convolution states | Unchanged; full prefill retained | **No** |
| Later 8 GQA Query, gate and output projections | Each layer distinct | Retained, still executed | **No** |
| Later 8 GQA K/V projections | Each layer distinct | Bypassed at runtime by one H32-derived bridge | Leave weights initially, for teacher/fallback |
| Later 8 GQA KV caches | Eight independent histories | One producer at block 36, seven later readers reuse | **These separate cache writes are removed** |
| Model MLP, norms, RoPE, residuals and LM head | Pretrained native Qwen | Retained | **No** |
| Vision tower | Present in official multimodal checkpoint | Left intact; text-only evaluation first | **No** |

**Important:** bypassing eight K/V projections and avoiding seven persistent KV caches does not remove the corresponding original K/V weight tensors from HBM. The SharedQwenAttention wrapper retains original modules so the checkpoint is reversible; only *KV cache* bytes are predicted to shrink. Consider optional pruning only after real H100 loading, quality checks, serialization/reload checks and a separately approved benchmark.

## 2. Actual mathematical rewiring

For a late GQA reader l and input state H_(l-1), original Qwen generates its own Q, K and V from the reader input, with pre-attention RMSNorm, per-head Q/K normalization, partial RoPE on Q/K, a query-side sigmoid gate and a post-attention output projection.

B1b defines E_t = H_(32,t) and computes exactly one bridge output per processed position:

~~~text
H32_t -- K_phi / KNorm + position RoPE --> shared K_t
      -- V_phi --------------------------> shared V_t

H_l-1,t -- original Q_l + q_norm + partial RoPE --> Q_l,t
Q_l,t + shared K/V --> causal dense SDPA --> sigmoid(query-gate) --> original O_l --> residual + MLP
~~~

The bridge writes its K/V only to the DynamicCache at **zero-based index 35 (one-based layer 36)**. Consumer layers 40, 44, 48, 52, 56, 60 and 64 obtain the same K/V tensor view, while using different Q and output projections. Layer 32 is the **source of H32**, not the KV producer; layer 36 is the **physical KV-cache producer**.

The first 32 blocks and later 24 GDN blocks keep their own native recurrent/conv states and full historical computation. B1b is a cross-layer sharing experiment, **not a decoder-prefill shortcut**, and uses dense PyTorch scaled-dot-product attention, **not CSA2 sparse top-k**.

## 3. Source interfaces to freeze and verify

Official Qwen source: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py

- The official model class is Qwen3_5ForConditionalGeneration; language blocks are at model.model.language_model.layers.
- Qwen3_5DecoderLayer dispatches between linear_attention (GDN) and full_attention (GQA). Only GQA blocks are rerouted.
- Qwen3_5Attention.q_proj jointly produces Query and output gate. Keep its q_norm, scaling, o_proj, native position_embeddings, and causal mask. The key is normalized and partly RoPE-rotated; the value is not.
- DynamicCache(config=model.config) provides per-layer storage; the adapter persists global K/V at layer index 35. This is **not guaranteed** compatible with StaticCache, beam reordering, packed variable-length input, generic multimodal position masks, torch.compile, tensor/pipeline parallel or offload.
- The pretrained checkpoint is **not functionally equivalent after replacing late K/V**. A new bridge needs adaptation training. No prior training step or feature-MSE score proves equal reasoning ability.

## 4. H100 verification — pass/fail gates

1. **Config:** pin Qwen model revision and transformers version; assert text config 64 layers, 48 linear and 16 GQA, hidden width 5120, head size 256, 4 KV heads; inspect actual module names.
2. **B0:** load original BF16 model, no CPU/disk offload, record GPU VRAM. Verify baseline cached forward and generation; do not patch first.
3. **B1b forward:** patch only one shared bridge, use text-only prompt, test full-sequence logits vs token-by-token cached logits on the **same patched weights** with sensible BF16 tolerance. Assert no future leakage.
4. **Cache physical identity:** 16 native groups vs nine in B1b; confirm only indices 3,7,11,15,19,23,27,31,35 have populated GQA caches, while 48 GDN recurrent/conv states still update.
5. **Training:** freeze backbone, run tiny 128-token gradient and optimizer step for bridge; record nonzero finite gradient, peak VRAM, model revision, actual loss, optimizer and checkpoint reload. H100 80GB is a target, **not a guarantee of sufficient activation memory**.
6. **Quality:** compare held-out token CE against native B0; for a causal architecture comparison add B0T with matched data/training budget. Quality retention is an empirical question.
7. **System claims:** separate unique KV bytes, GDN state bytes, peak allocated/reserved VRAM, prefill latency and decode tok/s. B1b's predicted 43.75% saving applies only to the **GQA KV component**, not all GPU memory.

Run exact commands in docs/H100_COMMANDS.md, beginning with model_probe.py, h100_preflight.py, pytest, benchmark.py for B0, a small feature projector warmup, and benchmark.py for B1b. Only use an actual authorized H100 host. Save real outputs and exit codes; do not report an unexecuted test as passing.

## 5. Main compatibility risks and choices

- **Representation mismatch:** pretraining taught each late GQA layer to use its own K/V from a different hidden depth; a shared H32 projector changes the function. Feature matching does not guarantee low perplexity. If loss is large, test layer-specific low-rank *read-time* transformations as a separate ablation, with added cost included.
- **GDN boundary states:** B1b retains all 24 late GDN state updates; attempting true CED prefill skipping without reconstruction would break recurrent-state semantics. Do not delete those layers or silently substitute zero states.
- **Memory/training:** original BF16 27.8B weights alone require roughly 51.8 GiB before runtime, activations, caches and optimizer. Frozen-backbone gradients through late layers may OOM on one H100; reduce sequence length or introduce verified activation checkpointing as a **tracked engineering change**, not an unreported baseline difference.
- **RoPE/masks:** position IDs and MRoPE must be obtained from native Qwen. Do not substitute independent layer position encodings or mistake B1's per-token appended KV table for a fixed-size recurrent state.
- **User-visible status:** mock tests passed; real H100 baseline, patched forward, grad, long-context and benchmark evaluations are pending. A source commit does not start GPU training.

## 6. Verdict

**Architecture structurally portable for B1b, with real correctness and memory gates still unverified. No Transformer/GDN layer deletion is required or justified. The only intended removal is seven redundant *persistent late KV histories*, while eight late native K/V projections are bypassed rather than pruned.**

Project: https://github.com/Yunbo-max/Testing-NMI
