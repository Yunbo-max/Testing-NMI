# Two-day wall-clock execution plan — one H100 80GB

## Definition of "done in two days"
By T+48 h: (1) original 27B loads in the intended runtime; (2) frozen baseline and deterministic samples are recorded; (3) a real 27B B1 shared-global-KV forward/decode path exists, with own GDN states maintained; (4) cache correctness and causal integrity tests pass; (5) a reduced trainable adapter performs a successful finite-loss backward/optimizer step, records checkpoint/resume logs, and initial training is ready to run or has started. Full scientific confirmation, convergence, sparse selector integration, and B2 GDN-state reconstruction are expressly **not** asserted.

If H100 model/runtime does not fit or kernels are unsupported, still finish bounded CPU / smaller-hybrid reference and publish an exact `blocked` receipt; never fabricate 27B success.

## Team and GPU coordination
- Coordinator: single integration writer, task ledger, hours and evidence receipts. Assign source reader + core code author + validation independent reviewer + training/eval preparer. Parallelize code/document/data work, **serialize GPU ownership** on one H100; no two giant model replicas.
- Use Research Autopilot Human–AI policy: pinned citations and code revision, source-backed mathematical assumptions, falsifiable predictions, matched baselines, native evaluation and negative-result preservation. Only human decides consequential changes of direction or exceeding budget.

| Window | GPU owner | Engineering work | Exit artifact / acceptance |
|---|---|---|---|
| H0–H3 | Infra | `nvidia-smi`, disk+VRAM+driver+Cuda, pin HF/config/transformers snapshot, launch unmodified 27B | `preflight.json`, frozen environment, model load log |
| H3–H8 | Baseline | Reproducible original Qwen, cached vs uncached parity, 256/1K/4K contexts, attention/GDN state accounting | B0 loss sample, memory snapshot, TTFT & decode starter traces |
| H8–H15 | Unit tests / code | Tiny CED/CSA2 reference and shared-table semantics; implement H32 capture + shared KV writer/readers in isolated patch | Toy parity/causality test; feature-cache shape/assertions, code review |
| H15–H26 | 27B integrator | Integrate patched `Qwen3_5Attention` readers at layers 36..64; preserve GDN; gated baseline fallback. Short-context forward + increment decode | `--mode original` within tolerance; patched mode no shape/NAN/casuality bugs |
| H26–H34 | Benchmark | Verify physical K/V storage sharing (not copied 8×); 1K/8K/32K load; compare kernel timings/VRAM; diagnose OOM | B0 vs B1 pre-training correctness/cost sheet, warmup excluded |
| H34–H41 | Training | Frozen source H32 extraction, per-layer KV target hook, trainable projector initialized, optimizer config, resumed checkpoint | One bounded feature-regression training smoke; finite/active gradients |
| H41–H48 | Independent reviewer + coordinator | Check end-to-end CE backprop, small teacher/student KL, independent reproduction, freeze configs/splits | A valid B1 training launch with step-0 checkpoint or explicit measured blocker |

## Gates / failover
**G0 Environment at H3:** BF16 weight load, H100 usable, host free disk >=150GB (200GB preferred), torch/transformers kernel smoke. If H100 BF16 fails, 8-bit inference preflight only; do not substitute quantized performance for BF16 baseline without rebenchmarking.

**G1 Parity at H8:** unmodified baseline repeated deterministic logits/generation. Use tolerances appropriate to exact kernel and BF16; log max abs/relative error. No downstream speed comparisons if baseline itself is broken.

**G2 Correctness at H26:** full-prefill-vs-incremental B1 (short causal test) within recorded tolerance; future-token perturbation does not alter earlier logits; correct cache positions; no duplicate/global KV cloning. Toy test is necessary but does not replace 27B gate. Treat before-training quality degradation as *expected* but log its magnitude.

**G3 Fit at H34:** actual peak VRAM within device; B0 and B1 run same runtime, same dtype and same warmup; measured unique K/V buffers agree with predicted 9 groups. Theoretical KV savings are not evidence of runtime savings until verified.

**G4 Training at H48:** trainable params explicitly enumerated, frozen params unchanged, gradient norms finite and nonzero, loss decreases on tiny overfit control, optimizer stepped and checkpoint reload produces consistent model state. Then begin the first real pilot. If G4 fails, **do not** launch an uncontrolled long run; keep diagnostics running and record cause.

## Later experiments (not a two-day claim)
- First scientifically meaningful training pilot: 50k–200k training tokens, 1–2k seq, gradually increase to 4k, measured rather than promised GPU-hours. Distill the projector then attention query/output adapters. Freeze B0 and a matched B0+LoRA control. Profile after 100–300 successful steps.
- Confirm on held-out language + code + long-range retrieval with matched decoding, end-to-end speed and HBM/host-cache metrics. Avoid adding sparsity before demonstrating shared-KV alone.
- B2 reconstruct late GDN states from causal H32 sequences and test bounded-replay error/cost in small hybrid model first. Only escalate to 27B on proof-of-feasibility.

## Not supported
No GitHub Actions, no workflow_dispatch, no pretending that a README triggers recurring GPU tasks. Research Autopilot checkpoint records and logs are directly pushed; GPU run is a real external persistent job/service, verified by process identity, stderr/stdout and checkpoint reads.
