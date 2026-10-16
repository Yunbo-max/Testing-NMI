# Testing-NMI — Qwen3.8-27B × DeepSeek-V4.1-Flash CED / CSA2

**Research question:** Can an encoder-derived, cross-layer shared KV memory reduce **real** GQA cache storage in an already-hybrid Qwen model **without unacceptable quality loss**? Can a later, separately tested state reconstructor unlock *actual* CED-style decoder-prefill skipping?

> **中文概述：** 从 Qwen3.8-27B 的 32 层边界表示产生一份可共享的全局 KV，让后 8 个 GQA 层复用，同时保留 24 个后半段 Gated DeltaNet 的状态更新。首先通过严格对照测出 KV/精度/速度的 Pareto 曲线，再研究是否能恢复 Decoder 的递归状态、跳过历史 Prefill。**B1 只是跨层共享 KV，尚不是完整 CED 或 CSA2。**

**Status (2026-10-16):** B1b source authored and small CPU/mock tests previously reported; no verified Qwen3.8-27B/H100 run, real optimizer evidence, native benchmark score, full CSA2 indexer, or B2 decoder-state reconstruction in this repository. The new data/NLL helpers also require first use on the actual host. `artifacts/` and private training corpora are not committed. GitHub is source/evidence storage, never the experiment executor.

## Research idea: from the paper gap to a testable design

The DeepSeek report combines **CED** (separate production of decoder global memory from expensive per-token deep decoder processing) with **CSA2** (compressed sparse retrieval, KV/index sharing). Earlier work such as [YOCO](https://arxiv.org/abs/2405.05254) and [Cross-Layer Attention](https://arxiv.org/abs/2405.12981) already studies shared historical KV. Our research question is **not** whether KV sharing was invented here, nor whether a smaller cache automatically means faster generation. It is whether that memory organization can be *transferred to Qwen3.8's hybrid Gated DeltaNet + GQA backbone* at a measurable quality/cost operating point.

**Qwen3.8-27B text backbone:** 64 blocks (48 Gated DeltaNet, 16 GQA). Split at the **output of block 32** (one-based). Blocks 36, 40, 44, 48, 52, 56, 60, 64 are eight later GQA **readers**. The first such reader (block 36) is the **physical cache producer**; the new projection *takes its input from H32*, not from the old block-32 internal KV. All 64 blocks, including 24 later GDN blocks, still run in B1b. Qwen's original Q-gate, Q/K normalization, partial RoPE, output projections and FFNs stay intact. B1b currently uses **dense PyTorch SDPA**, not an official DeepSeek CSA2 kernel.

### Mathematical derivation

Let \(H_{\ell,t}\) be the representation of token \(t\) after layer \(\ell\). For every original late GQA reader \(\ell\in\mathcal R=\{36,40,44,48,52,56,60,64\}\), the ordinary cache contains layer-dependent values (schematically):

\[
K_{\ell,t}=\operatorname{RoPE}\!\left(\operatorname{KNorm}_{\ell}(W^K_{\ell}\operatorname{LN}_{\ell}(H_{\ell-1,t}))\right),\qquad
V_{\ell,t}=W^V_{\ell}\operatorname{LN}_{\ell}(H_{\ell-1,t}).
\]

**B1b hypothesis:** replace those eight persistent late K/V caches by one trainable projection of the encoder boundary \(E_t=H_{32,t}\):

\[
\tilde K_t=\operatorname{RoPE}\!\left(\operatorname{KNorm}_{\phi}(W^K_{\phi}E_t)\right),\qquad
\tilde V_t=W^V_{\phi}E_t,\qquad
M_t=M_{t-1}\mathbin{\|}(\tilde K_t,\tilde V_t).
\]

Here \(\|\) is **append along time**, not a fixed-size linear-attention recurrent update and not aggregation of all 32 encoder-layer KV caches. Each late reader keeps its own \(Q_{\ell,t}\), attention weights, output gate and FFN while reading the shared \(M_t\). GDN recurrent/conv states remain per-layer.

**What can be lost?** Prior to training, extract the eight original **pre-RoPE** (K-normalized) K/V teacher targets \(Y_{\ell,t}\) at the same token positions. Any single shared predictor \(g(E_t)\) has the exact per-sample identity:

\[
\frac1{|\mathcal R|}\sum_{\ell\in\mathcal R}\|g(E_t)-Y_{\ell,t}\|_2^2
=\underbrace{\|g(E_t)-\bar Y_t\|_2^2}_{\text{learnable approximation}}
+\underbrace{\frac1{|\mathcal R|}\sum_{\ell\in\mathcal R}\|Y_{\ell,t}-\bar Y_t\|_2^2}_{\text{unavoidable cross-layer variance}},\quad
\bar Y_t=\frac1{|\mathcal R|}\sum_{\ell}Y_{\ell,t}.
\]

The second term is a **lower bound on direct layerwise feature matching error** for a single identical shared vector; it is **not** a lower bound on end-task perplexity. Large variance predicts difficulty for the feature warmup, *not necessarily* failure of the model after end-to-end adaptation. An independent future candidate could use per-reader transformations at read time, but adds parameters/compute and needs a new matched control.

**Why this does not yet reproduce full CED:** GDN has a recurrent state \(S_{\ell,t}=U_{\ell}(S_{\ell,t-1},H_{\ell-1,t})\) plus convolution state. Sharing later GQA KV does **not** yield the missing late-GDN boundary states if earlier decoder history is skipped. B2 will only claim CED-style prefill savings if it actually reconstructs those states with measured accuracy and *net* compute savings.

### Pre-registered predictions and falsifiers

| ID | Prediction / null alternative | Required observation that would reject the claim |
|---|---|---|
| H1 (storage) | Real B1b holds 9 distinct BF16 GQA KV groups, not 16 | Extra persistent late K/V replicas or actual counted group/byte mismatch |
| H2 (quality) | After **matched** adaptation B1b has tolerable held-out CE/task loss versus B0T | Document-paired held-out degradation beyond registered tolerance; report even if it fails |
| H3 (latency) | Fewer K/V bytes *may* improve decode or memory-limited throughput | Same-host paired timing shows no gain or a regression (cache claim may still hold) |
| H4 (B2 only) | Reconstructed GDN states might permit decoder historical skip | Reconstruction + inference costs >= original prefill, or long-continuation quality collapses |

**KV cost calculation (prediction only):** one BF16 GQA group uses \(2\times4\times256\times2=4096\) bytes/token. Original 16 groups: **65,536 bytes/token**. B1b: eight early groups + one late shared group = **36,864 bytes/token**, a **43.75% reduction of the GQA KV component**; at 32,768 tokens that is 2.0 GiB vs 1.125 GiB. This is **not** a 43.75% reduction in total VRAM, GDN state, or prefill FLOPs.

## Experimental arms and implementation status

| Arm | Modification | Purpose | Status |
|---|---|---|---|
| **B0** | Unmodified Qwen3.8-27B, no further training | Native quality/systems baseline | Benchmark command authored; H100 unverified |
| **B0T** | Unmodified Qwen + **matched** trainable adapter/tokens | Control for extra optimization and parameters | Planned, **not implemented** |
| **B1a** | Four late KV groups (pairwise sharing) | Compression-quality curve | Planned |
| **B1b** | One H32-derived KV shared across eight late GQA | Primary adapted architecture | Source + CPU mock tests; 27B H100 unverified |
| **B1q** | Independently cached KV at comparable storage budget, e.g. quantized | Strong budget-matched memory comparator | Planned |
| **B1s** | B1b plus actual sparse indexer and index reuse | Test CSA2 retrieval separately | Planned, **not CSA2 yet** |
| **B2** | Decoder prefill skipping + learned GDN state reconstructor | Full CED-style acceleration research | Hypothesis only |

A result comparing trained B1b to *untrained* B0 cannot establish a quality advantage. **B0T is mandatory for any architecture-induced quality claim.** The B1b feature-warmup and other auxiliary compute must be included in training-budget accounting. Fixed tokens alone do not guarantee equal FLOPs or GPU hours: report both.

## Dataset plan: exact sources, splits and budgets

**Main adaptation corpus (selected):** [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), configuration **sample-10BT**, upstream **train** split, streamed in a fixed shuffled order. Its public licence is ODC-BY; respect attribution and source terms. This project uses **text-only causal LM**, not image/video evaluation. Neither the base model's pretraining membership nor its training-contamination status is known, so we claim *post-adaptation split separation*, not that any text is globally unseen by Qwen.

Use `scripts/prepare_fineweb.py` to make private JSONL files accepted by the existing `scripts/train_27b.py`. It pins both the dataset and tokenizer revisions, shuffles with seed **1337**, deduplicates **exact text**, then assigns each document by SHA256-text hash modulo 10: **80% train / 10% validation / 10% internal test**. These are our **constructed within-source splits**, *not* official FineWeb-Edu validation/test splits. The manifest stores revisions, actual token/document counts and per-file SHA256; near-duplicate detection is **not** yet implemented.

| Stage | Training exposure target | Internal validation / test | What it can establish |
|---|---:|---|---|
| Unit smoke | `data/smoke.jsonl` only | none | Code paths, **not model quality** |
| H100 start | 16 optimizer steps, seq 128, accum 1, batch 1 (actual <= ~2k training tokens) | Current `train_27b.py` checks a **4-example** val slice | Real backward/optimizer/checkpoint only |
| Pilot data **prepared** | **120k** train bounded tokens; **20k** val, **20k** internal test | Frozen, disjoint documents | Data available; *not* proof these tokens were trained on |
| Controlled adaptation | 100k then 200k **actually consumed** train tokens, paired B0T/B1b | Fixed val/test, multiple seeds as budget allows | Preliminary relative CE/quality |
| Stronger study | ≥1–2M actually consumed tokens if H100 timing and loss curves permit | Native benchmarks + independent confirmation | Candidate quality–cache Pareto curve |

**Important:** the present trainer cycles input JSONL and truncates each sample to `--seq-len` without packing; do not infer actual token exposure from the size of the prepared file. Use `tokens_seen` in its log, count repeated exposure, and do not call 16 smoke steps “120k-token training.” Before larger comparisons implement/qualify B0T and a consistent data sampler/packing policy, or freeze the **same** no-packing data order for every arm.

Prepare data (on the authorised host, with network and exact immutable revisions):

```bash
python -m pip install -e '.[train,test]'
python scripts/model_probe.py --out results/model_probe.json
export MODEL_REVISION=$(python -c 'import json;print(json.load(open("results/model_probe.json"))["revision"])')
export DATA_REVISION=$(python -c 'from huggingface_hub import HfApi;print(HfApi().dataset_info("HuggingFaceFW/fineweb-edu").sha)')
python scripts/prepare_fineweb.py --model-revision "$MODEL_REVISION" --dataset-revision "$DATA_REVISION" \
  --train-tokens 120000 --val-tokens 20000 --test-tokens 20000 \
  --seq-len 128 --seed 1337 --out artifacts/data/fineweb_pilot_v1
export TRAIN_JSONL="$PWD/artifacts/data/fineweb_pilot_v1/train.jsonl"
export VAL_JSONL="$PWD/artifacts/data/fineweb_pilot_v1/val.jsonl"
```

**Independent language-model diagnostic:** [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext), config `wikitext-103-raw-v1`, original `validation` for development and `test` only once for frozen confirmation. The local NLL helper can evaluate an exported JSONL of that corpus but **its per-document `--seq-len` truncation is NOT the official standard WikiText-103 PPL**. Full-corpus/sliding-window scoring must be implemented and verified before reporting a “WikiText-103 PPL” headline.

**Other public benchmarks reserved for evaluation only (never included in our adaptation data):** [RULER](https://github.com/NVIDIA/RULER) (synthetic 4k/8k/16k/32k retrieval, tracing, aggregation), [LongBench-v2](https://github.com/THUDM/LongBench) (503 multiple-choice questions; run only instances that fit and label a restricted subset, never silently truncate), [openai/human-eval](https://github.com/openai/human-eval) (164 code problems, pass@1 under sandboxed execution), and [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) (test-split math final-answer exact match). **Benchmark runners/scorers for these four suites are not implemented in this repository yet; no scores are claimed.** Pin evaluator/data revisions when qualified.

## Evaluation: exactly how claims will be measured

**Primary quality:** token-weighted next-token held-out cross entropy in nats/token, \(\mathrm{CE}=-(1/N)\sum_{i=1}^{N}\log p(x_i\mid x_{<i})\); **PPL** \(=\exp(\mathrm{CE})\). Compare B1b with B0T on *identical tokenized positions* and report \(\Delta\mathrm{CE}=\mathrm{CE}_{B1b}-\mathrm{CE}_{B0T}\) plus relative change. The current 5% **relative CE increase** is an early warning threshold, not a confidence interval or proof of equivalence. Use document-level paired bootstrap intervals for the metric difference when the sample size is large enough, and independently report run-to-run seed variation.

**Current executable local evaluation:** `scripts/eval_nll.py` computes token-weighted CE/PPL for a fixed JSONL slice with separately reset, truncated documents; it records the input file hash, exact tokenizer/model revision, number of predicted tokens, and limitations. Evaluate B0/B1b sequentially on one GPU. This is a **bounded-document CE diagnostic**, not native LongBench/RULER/WikiText scoring.

```bash
python scripts/eval_nll.py --arm base --revision "$MODEL_REVISION" \
  --data "$VAL_JSONL" --seq-len 128 --max-examples 64 \
  --out results/runs/B0_val_nll.json
# after actual training, use the exact *observed* adapter checkpoint path:
python scripts/eval_nll.py --arm shared --revision "$MODEL_REVISION" \
  --adapter artifacts/train_b1/adapter-step-000016.pt \
  --data "$VAL_JSONL" --seq-len 128 --max-examples 64 \
  --out results/runs/B1b_val_nll.json
```

This particular **B0 vs B1b** command is diagnostic; **do not claim matched training** until B0T is implemented. Always freeze train/val/test lists *before* using validation to choose the adapter.

| Question | Metrics and frozen protocol | Readiness |
|---|---|---|
| Does global-KV sharing physically happen? | Unique GQA K/V tensors, bytes/token, eight-reader producer writes once; verify GDN states unchanged | Mock-tested; actual 27B pending |
| Does causal generation still work? | Full-prefix vs incremental logits with proper cache, future-token perturbation, positional offset, grad/finite-loss | CPU tests; 27B pending |
| Does quality recover? | CE/PPL on fixed FineWeb internal val and external WikiText data; B0T matched adaptation, train tokens/FLOPs and parameter count | **Bounded CE helper available**; B0T/full benchmark pending |
| Does long-history retrieval survive? | RULER native exact-match by task and context at 4k/8k/16k/32k; depth-controlled retrieval/tracing | Native evaluator pending |
| Does long-form understanding survive? | LongBench-v2 MCQ exact accuracy by domain/length, documented full vs feasible subset | Native evaluator pending |
| Does code/math capability regress? | HumanEval pass@1 (one deterministic completion/task, sandbox) and GSM8K test exact numeric answers | Native evaluators pending |
| Does runtime improve? | Measured GQA bytes, **separate GDN state**, peak allocated+reserved VRAM, prefill forward latency, decode tok/s at 1k/8k/32k | Initial benchmark command available; repeat harness pending |

**Timing protocol:** same single H100, identical model revision/dtype/backend, prompt IDs and output lengths; no MTP/speculative decoding; warm up 3 times then ≥10 paired repetitions, alternate B0/B1 order, report median, p10/p90 or bootstrap confidence interval, startup vs prefill and decode separately. The existing `scripts/benchmark.py` is a **single-run diagnostic**, and its `prefill_seconds` includes forward processing without a full serving-stack TTFT definition. Its output doesn't qualify a speed claim by itself. Record total wall time, measured GPU use and quality alongside speed.

**Hardware gate:** 27.8B BF16 weights alone are roughly 51.8 GiB, excluding model activations/optimizer/caches. Use frozen BF16 backbone + FP32-master projector, batch 1, seq 128 for first optimizer step; do not claim full-parameter training fits one H100. If OOM, record the error and re-evaluate the design rather than hiding CPU offload or changing baselines inconsistently.

## Reproduction commands and evidence lifecycle

```bash
# GitHub is only source + evidence storage; runs happen on the real H100 host.
git clone https://github.com/Yunbo-max/Testing-NMI.git
cd Testing-NMI
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[train,test]'
python -m pytest -q tests
python scripts/h100_preflight.py --out results/preflight_h100.json
python scripts/model_probe.py --out results/model_probe.json
# Prepare the pinned data above before starting the following:
bash scripts/run_h100.sh
```

See [`docs/H100_COMMANDS.md`](docs/H100_COMMANDS.md) for stepwise execution, [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md) for quality and fairness controls, [`configs/experiment_matrix.yaml`](configs/experiment_matrix.yaml) for the status/config matrix, [`docs/RESEARCH_AUTOPILOT.md`](docs/RESEARCH_AUTOPILOT.md) for the research workflow, and [`agents/AGENT_TASKS.json`](agents/AGENT_TASKS.json) for dependent work packages.

**Autopilot handoff rule:** source-backed research question → mathematical object and derivation → condition-specific prediction and falsifier → qualified B0/B0T and strong alternatives → actual experiment → independent evidence review → next decision. Track exact git SHA, Qwen revision, dataset revision+SHA256, optimizer state, trainable parameters, *actually consumed* tokens, seed, GPU characteristics, command/exit status, raw metrics, and negative results. **Do not** claim H100 runs from CPU tests or source commits. **No** `.github/workflows`, GitHub Actions or `workflow_dispatch`.

## Primary sources

- [DeepSeek-V4.1-Flash technical report](https://arxiv.org/html/2609.19969) and [official implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/main/inference)
- [Qwen3.8-27B checkpoint](https://huggingface.co/Qwen/Qwen3.8-27B) and [Hugging Face Qwen3.5 modeling implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
- [YOCO](https://arxiv.org/abs/2405.05254), [CLA](https://arxiv.org/abs/2405.12981), [RULER](https://github.com/NVIDIA/RULER), [LongBench-v2](https://github.com/THUDM/LongBench)
- [Research Autopilot Human–AI source](https://github.com/Yunbo-max/Research_Autopilot)
