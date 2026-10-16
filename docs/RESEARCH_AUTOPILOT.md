# Research Autopilot: controlled B1 architecture adaptation

## Parent goal and target

Original user goal: reproduce relevant DeepSeek-V4.1-Flash architectural mechanisms starting with **Qwen3.8-27B**, with a real H100, complete code/commands on `Testing-NMI`, objective ablations and a 48-hour pilot culminating in first training. **The 48 hours start once H100 host access, data and pinned model revision are available**; writing repository content does not start that clock.

The target here is an executable **B1b** cross-layer global KV sharing adaptation, not full CED. Planned later tasks: CSA2 sparsity and Gated DeltaNet decoder state reconstruction.

## Mathematical object

Qwen3.8 has 64 language layers. Let `E_t = H_{32,t}` denote post-Encoder boundary representation; `R={36,40,44,48,52,56,60,64}` are the late GQA layers (1-indexed). In the original model each reader `l` computes own `K_{l,t},V_{l,t}` from its layer input, while 24 late GDN layers maintain independent recurrent states `S_{l,t}`.

B1b introduces `G_phi(E_t)=(k^G_t,v^G_t)`, one fixed-size vector pair per token, shared by all GQA readers. Each reader retains its original `q_l(h_{l-1,t})`, output gate, `W_l^O`, feed-forward and DeltaNet layers. The operation is one shared *cache table* `M_t=Append(M_{t-1},G_phi(E_t))`, not one fixed-size linear-attention state. Reader Attention remains **dense SDPA**, not indexed CSA2.

Full CED's prefill skip would require computing each late GDN initial state at prompt end. With `S_{l,t}=U_l(S_{l,t-1},h_{l-1,t})`, future state depends on the omitted historical decoder features. Thus even an exact shared GQA global KV does NOT justify skipping decoder-GDN prefill. This is a causal dependency argument, not an impossibility theorem: a cheaper approximate state reconstructor might be trained, and its error/cost must be tested.

## Predictions and falsifiers

**H1: cache storage:** With all else held equal, B1b should have 9 (8 early + 1 late) physically allocated GQA KV groups rather than 16. In BF16 it predicts 36,864 vs 65,536 GQA bytes/token (43.75% reduction). **Falsifier:** native cache retains other late independent tensors, or measured bytes differ after excluding GDN/misc, requiring implementation fix before any resource claim.

**H2: accuracy:** Frozen-base random B1b may degrade; projector feature warmup and CE should recover some accuracy. **Falsifier of a strong quality-preservation claim:** matched trained B0T beats trained B1b on held-out native NLL and pre-registered long-context tasks beyond the tolerance defined in the evaluation protocol. Always retain unfavorable results.

**H3: performance:** Cache savings may not yield equal-speed improvement. Every extra projector and altered SDPA kernel has runtime cost. Compare cold/warm TTFT, decode t/s, peak/reserved memory and full latency with repeated measurements. **Falsifier of a speed claim:** B1b is slower at matched workload; retaining a cache-saving claim then does not imply faster decoding.

**H4: true CED prefill skip (later B2):** If a state reconstructor can produce `S_hat` from encoder-side information cheaply and with bounded generation error, then skipping most late GDN historical computation may become feasible. **Falsifier:** state reconstruction cost ≥ original late prefill or controlled downstream accuracy collapses. No B2 claim without actual GPU implementation and measurement.

## Baselines and evaluations

- B0: unmodified frozen Qwen3.8-27B; measured memory + loss. This is the engineering anchor, NOT fair evidence of architecture-specific post-training gain.
- B0T: Qwen under an equally budgeted adapter (same data, tokens, comparable optimizer/parameters/tuning) to isolate extra-training effects. This is a **required future implementation** for strong quality claims.
- B1a: pair-share among the late 8 GQA layers, source projections fixed/trained; **planned**.
- B1b: late 8-reader shared K/V, **implemented locally and CPU-mock-tested; GPU unverified**.
- B1q: ordinary per-layer KV quantization with similar bytes; **planned**, stronger memory-budget comparator.
- B1s: true top-K sparse attention using real selected indices; **planned**; must separately measure indexing overhead and errors.
- B2: encoder-derived Decoder GDN state reconstruction + verified full prefill skip; **research lead only**.

Held-out LM cross entropy/perplexity is required; additional validation suite should include length-controlled retrieval (standard RULER/LongBench protocols), code and long autoregressive constraint consistency with frozen task seeds. Native benchmark semantics and license controls apply. A toy diagnostic, custom prompt or seed search cannot replace native scoring.

## Fixed experimental principles

Train and validation JSONL **different files**, no data leakage, pinned tokenizer/model/HF/torch revisions, identical prompt and output lengths and measured token counts. Isolate B0/B1 on one GPU so models never compete for VRAM, warm kernels before timing, alternate run order across repeats, report confidence/variation and all failures. Report **GQA bytes, GDN state, peak allocated, TTFT, prefill t/s and decode t/s separately**. Record actual GPU model and CUDA/PyTorch/toolchain revisions.

First pilot acceptance is only: real Qwen text model loaded, correct 64-layer source mapping, one producer cache & no late duplicate, valid full-vs-incremental (within numerical tolerance), finite nonzero projector gradient, optimizer step, loss trace, adapter checkpoint plus restoration. A successful pilot **does not** prove no capability loss or CED speedup.

## Autopilot cycle/stop policy

Use the `main` Human–AI research mode. Read closest primary papers and code; read actual results before selecting next experiment. Engineering bugs can be repaired automatically, substantive direction and larger GPU investment require human decision. Preserve negative results and unknown statuses. Before expanding data/compute, test failure hypotheses on a discriminating measurement. Never replace an uncompleted benchmark with repeated toy tests. Store actual commands, logs, source revision, failure trace and next action. GitHub stores files only, never executes GPU jobs, and no GitHub Actions/workflows are allowed.
