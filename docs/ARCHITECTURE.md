# Architecture transfer contract — not a drop-in 40→64 rewrite

## Original source of truth
- DeepSeek V4.1: 20 causal encoder / 20 causal decoder. From paper Eq. 1: `C_l = H_{L/2} W_l^{KV}`, `Z_l = H_{L/2} W_l^Z`. Decoder still has layer-specific query and local SWA KV. CSA2 modes: Full, Reindex, Reuse; Full DOES NOT mean dense full-context attention. See https://arxiv.org/html/2609.19969#S2.SS2 .
- Qwen3.8-27B: 64 blocks of `(GDN,GDN,GDN,GQA)`, width 5120. GQA 24 Q heads, 4 KV heads, head dim 256, partial RoPE; 48 other blocks store gated delta recurrent state (and short causal convolution state), not normal per-token KV. Model spec at https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json .

## Reproduction tiers
A. **Tiny faithful CED/CSA2 reference** (no pretrained weights): 40 or fewer configurable blocks with Encoder-derived decoder main KV, local-window branch, causal masking, index Full/Reindex/Reuse, prefix vs incremental parity; this addresses *algorithmic fidelity*.
B. **Qwen 27B adapted shared-KV B1** (first 48 hours): compute H32, construct shared global K/V, let late eight *GQA* readers query that same data; maintain original late GDN states, compute full 64 layers for each new token and full GDN prefill. This is **cross-layer sharing adaptation**, not full CED.
C. **Qwen 27B CED state reconstruction B2** (subsequent research): hypothesize a cheap causal map `R_l(H32[1:t]) -> S_l(t)` that reconstructs each later GDN recurrent/conv state at prompt boundary without expensive late FFN runs. Requires separate proof/quality/perf evaluation; cannot assume possible or easy.
D. **CSA2 exact index semantics** (separate): main-KV indexing (`Full, Reindex, Reuse`), top-K candidate pooling, causal validity and location mapping, sparsity kernel, while distinguishing sparse attention from cross-layer KV share.

## Exactly what B1 changes
Layer numbering: first 32 layers contain 8 GQA and 24 GDN; latter 32 likewise. Source `H32` is the **output** of layer 32. Shared K/V tensor has shape `[batch, seq, 4, 256]` for each of K and V (2×4×256=2048 scalar values per position). The new projector is e.g. `nn.Linear(5120, 2048, bias=False)` producing the two packed tensors. Add optional per-reader minimal gain/bias or rank adapters only if separately recorded in the ablation. Apply position encoding to K with the same *partial* rotary scheme that Qwen's GQA uses; V has no RoPE. Respect the actual HF code's Q-projection/gating, masked SDPA behavior, cache API and memory layout; do not mix pre-RoPE/after-RoPE K features when initializing from teacher.

Create the shared KV **once**, by processing all Encoder outputs for prefill, and then append one record for each *processed* new token in decode. Readers 36,40,44,48,52,56,60,64 use their own Q and attention output projection; they must not synthesize eight cloned caches. The 24 decoder-side GDN blocks still sequentially update their **own** states in B1, for full prompt and for generated tokens.

CAUSALITY: during training the global table may be constructed from full-sequence H32, but decoder read at position t MUST be masked to <=t. During decode preserve explicit absolute token positions; no double append; cache length and counts advance only when a new token is fed through the backbone. Packed/padded batches require sequence-specific mask validity.

## Initialization / trainability
Replacing original eight different GQA K,V with a single untrained projection **will change outputs**. Do not use this to claim immediate lossless compression. Begin by training the new shared projector with frozen teacher features, and then distill final logits. A strict control performs the same total training tokens without changing the K/V structure.

Feature warmup objective (diagnostic only):
`L_feat = (1/8) Σ_l [||Kshared - stopgrad(Kteacher,l)||² / var(Kteacher,l) + ||Vshared - stopgrad(Vteacher,l)||² / var(Vteacher,l)]`.
Collect K in a position convention consistent with the student (prefer pre-RoPE for stable per-layer comparison). All targets are on the **same fixed token distribution**. This objective may fail because layer-wise K/V are not mutually compatible; use the failure as evidence, not a bug workaround. A per-layer read transform improves expressivity but may add substantial compute; cost it explicitly and do not count on memory savings unless only shared latent is stored.

## Core dependency obstruction to full CED on hybrid Qwen
`S_{l,t} = U_l(S_{l,t-1}, h_{l-1,t})` for each GDN layer. If historical decoder hidden states are skipped during prefill, S_l at the generation boundary is unknown. A global GQA KV from H32 does not reconstruct this state. Any CED speed claim therefore must measure the computational cost and error of a **causal** state reconstructor for the 24 decoder GDN layers. Avoid deriving state from future prompt tokens relative to the target position. Without B2, count no DeepSeek-like half-prefill gain.

## Reference code warning
The DeepSeek paper architecture and official reference inference implementation are not automatically identical in every training/deployment path. Reproduction agent must trace paper Eq.1 against actual `Transformer.forward`, the KV source layer and the runtime prefill path, and record any missing branch. Do not assume copying model.py alone implements the full CED algorithm.
