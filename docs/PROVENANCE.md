# Source and implementation provenance (2026-10-16)

Observed through connected GitHub/Hugging Face connectors, not inferred from headlines.

- `Qwen/Qwen3.8-27B`, model architecture `qwen3_5`, ~27.8B parameters, Apache-2.0. Immutable HF commit **must be captured on execution** via scripts/model_probe.py (do not claim an unobserved pin here).
- Official Qwen text source: `src/transformers/models/qwen3_5/modeling_qwen3_5.py` in `huggingface/transformers`, observed file blob SHA `14a7571639ed09bc8dc998ab1f2fd1f32fdeb32e`. GitHub code-search observed commit `09281b8d56b08199ce9f397c694f38853cc45278`; use this exact snapshot, or pin and recheck later code before running.
- `Qwen3_5Attention` lines 735–807: `q_proj` produces query AND sigmoid gate; per-head `q_norm`, `k_norm`, partial RoPE, `cache.update`, attention interface then gated `o_proj`.
- `Qwen3_5DecoderLayer` lines 847–900: `linear_attention` Gated DeltaNet or `full_attention` GQA, each with its own FFN/residual.
- `Qwen3_5TextModel` lines 1208–1294: `embed_tokens`, `layers`, `norm`, `rotary_emb`, mask mapping; 4-axis (text/T/H/W) position handling.
- `Qwen3_5ForConditionalGeneration` lines 1752+: model `model.language_model.layers` and multimodal visual branch. This project initially tests **text-only** input, not full multimodal capability.
- HF `cache_utils.py` `DynamicLayer.update` lines 103–149 stores `.keys,.values`; `Cache.get_seq_length` lines 1532–1554 uses first non-linear attention layer as the length source. Project B1 writes late global KV once at zero-based layer index 35 and leaves 39..63 empty.
- DeepSeek tech report `https://arxiv.org/html/2609.19969`: CED, CSA2, state replay; project applies only cross-layer K/V projection sharing to hybrid Qwen during B1.
- Research Autopilot repo `https://github.com/Yunbo-max/Research_Autopilot`, `main` README title `Autopilot Human-AI 10.15`, policy v0.1.10 observed 2026-10-16. Code generation follows object-first inquiry and testable mathematical predictions; adopted plugin on H100 host must be verified separately.

**Known gaps:** no live HF package installed in this CPU container; no H100 connection; no real Qwen checkpoint verification; no native full CED reproduction; no CSA2 indexer kernel. CPU mock tests are valid as low-level diagnostics only. Logs must reflect this separation.
