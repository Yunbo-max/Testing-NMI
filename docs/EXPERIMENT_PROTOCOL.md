# Pre-registered experiment protocol — model / data / resource fairness

## Main causal question
Can layer-shared KV produced from H32 reduce **actually allocated** GQA KV storage on Qwen3.8-27B while preserving performance after an equal-data, equal-update adapter training regime?

H1: B1 cuts nominal GQA cache from 16 groups to 9 groups at BF16 (43.75%); if physical sharing is real, measured stored *GQA KV bytes* approach the theoretical ratio, excluding metadata/other caches. Not a claim about total VRAM or throughput.
H2: Under matched steps, compute/data, B1's held-out cross entropy and task pass rates are no worse than pre-registered allowable degradation relative to a trained LoRA control. This may fail, especially at long contexts.
H3 (B2 only): A causal state reconstructor can reduce full-prompt decoder work while matching native GDN recurrent/conv boundary states closely enough for stable continuation. This is *not* part of B1 and is a high-risk hypothesis.

## Frozen treatment arms
| Arm | Model | Training | Attention/cache | Purpose |
|---|---|---|---|---|
| B0 | Original Qwen3.8-27B | none | original all 16 GQA K/V and 48 GDN | untrained reference |
| B0T | Original Qwen3.8-27B | matched LoRA/adaptation tokens | original KV | controlled training-only comparator |
| B1a | Same checkpoint | matched training | each group of 2 late GQA layers share source-H32 K/V; 8 initial + 4 late groups = 12 GQA cache groups | compression sweep (25%) |
| B1b | Same checkpoint | matched training | all 8 late GQA layers share 1 KV group; total 9 | main treatment (43.75%) |
| B1q | Same checkpoint | none | original KV, quantized 8-bit if supported | qualified compression-only comparator |
| B1s | Same checkpoint | matched | shared KV + separate sparse selector | optional beyond first 48h; must measure index overhead |
| B2 | Same checkpoint | substantial retraining expected | full decoder prefill skip + reconstructed GDN states | future test, not in first-two-day claim |

If matched LoRA control cannot have exactly the same number of trainable parameters, report counts and build two controls: (a) same trainable modules except changed memory semantics; (b) closest trainable parameter budget. Don't call weak controls fair.

## Training data and splits
Pin tokenizer/packing, shuffle seed, data revisions/licenses. Initial 100k-token smoke from a clean permissive mixture: approximately 50% general prose, 25% code, 25% reasoning/math or long documents; labels are next token (causal CE). The proportions are an experimental proposal; if actual sources are unavailable, record substitutions instead of claiming these were used. Reserve group-disjoint documents for dev and held-out, no leakage. For meaningful results use substantially more data and equal token exposure/optimizer budgets across B0T/B1, recording measured throughput. Keep all reports for negative runs.

## Diagnostic tests (not public benchmark claims)
- 8, 128, 1k, 8k, 32k-token forward and incremental cache; EOS/padding; batch=1 then 2.
- Causality by perturbing token t+1..end and checking earlier output invariance; reset/drop cache between prompts.
- KV count: for GQA 16 × (2 × 4 × 256 × 2 BF16 bytes)=65,536 B/token. Late-8 shared => 9 × 4,096=36,864 B/token; 32k context: 2.0 GiB vs 1.125 GiB theoretical for GQA only.
- Cache key/rotary: compare 1-based GQA reader layers [36,40,44,48,52,56,60,64], position offsets and partial-RoPE dim 64. Confirm K no future leakage, V unrotated. No per-reader replicated storage.

## Public measures, after training
- Language: validation loss and perplexity on a pinned public held-out corpus, e.g. WikiText-103 val as quick measure plus a separate code-text validation set; identify exact versions and token counts.
- Long context: RULER / NIAH controlled diagnostic at 1k/4k/16k/32k contexts plus native LongBench-V2 when evaluator is qualified. Report native unchanged metrics separately. Long output cross-chapter consistency is a task-specific diagnostic, not native benchmark.
- Code: HumanEval pass@1 with executable tests on a held-out set, fixed decoding settings, not a hand-picked demo.
- Reasoning: fixed public math/evaluation subset, teacher-forced loss and solved rate separately; do not infer general superiority from 50 examples.
- Memory: peak HBM VRAM `torch.cuda.max_memory_allocated`, NVML allocated memory, unique K/V allocation bytes, GDN state, allocator cache, CPU/SSD persistent bytes.
- Speed: TTFT/prefill throughput, decode tokens/s, index overhead separately. At lengths 1k→256, 8k→256, 32k→512, and 64k→512 if feasible. Warm 3, record >=10 paired repeats, vary order. Same compiled kernel, batch, GPU and precision.
- Always include total token throughput, time-to-quality and peak memory **at matched quality**. A speedup while quality collapses is not a success.

## Exact reporting
Every row contains: run ID, treatment, git commit, model revision, environment, hardware memory/driver, input-token distribution, optimizer, max sequence length, trainable param count, tokens seen, wall-clock / GPU-hours, loss/logits metrics, cache bytes, TTFT, decode speed, seed, complete command and exit status, raw stdout/stderr location, checker/reviewer receipt.

## Discriminating refutation
1. Shared K/V buffers appear eight times: the memory claim fails; fix implementation.
2. Cache shrinks only theoretically but peak VRAM does not improve: inspect allocator/activation overhead; claim restricted to KV storage.
3. B1 after comparable train is worse than B0T by >5% **relative** in held-out CE, or catastrophic task regression: reject "negligible quality loss" at that condition; investigate stronger adapters and tradeoff, not silently move tests.
4. Prefill throughput not faster than B0: expected for B1 because 24 late GDN states still built; do not relabel this a CED speedup.
5. B2 reconstruction consumes similar cost to full decoder or state mismatch diverges on long prompts: CED-style prefill saving has not been shown for Qwen hybrid.

The 5% CE guardrail is a preregistered **screen**, not statistical evidence by itself; report absolute CE, PPL, confidence or paired document uncertainty and training-run variance before any broad claim.
