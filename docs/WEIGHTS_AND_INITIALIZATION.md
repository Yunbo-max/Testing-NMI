# Weight initialization contract: Qwen3.8-27B → shared-KV bridge

**Default: PRETRAINED TRANSFER, NOT TRAINING 27B FROM SCRATCH.** Pin the exact `Qwen/Qwen3.8-27B` model revision. All original 64 language blocks, 48 Gated DeltaNet blocks, GQA Q/output/gate modules, FFNs, embeddings and LM head keep pretrained weights. Freeze the backbone. The 8 original late GQA K/V projection **weight tensors remain loaded** as frozen modules for model compatibility, but the B1b forward path bypasses their K/V calculation and uses a single H32-derived shared cache, physically written at layer 36 (one-based).

## What is copied?

For native original layer 36 with pretrained attention weights and normalizer, the bridge is created with exactly matching shapes (H32 width 5120; each new K/V has 4 heads × 256 dims):

$$
W^K_{\phi,0} = W^K_{36}, \qquad
W^V_{\phi,0} = W^V_{36}, \qquad
\theta_{\phi,0}^{\mathrm{KNorm}} = \theta_{36}^{\mathrm{KNorm}}.
$$

**The output is not identical to original layer-36 KV**: the source layer receives its *own* normalized hidden input, while the bridge receives H32. Copying is a transferable initialization hypothesis, not a lossless mathematical equivalence. B1b preserves all 24 decoder-side GDN recurrent+conv state updates and all Q/O projections; no 64→32 layer deletion or CED decoder-prefill bypass occurs.

## Trainable modules and objective stages

| Stage | Actual code | Model in GPU | Trainable weights | Loss | Status |
|---|---|---|---|---|---|
| 0: Teacher | `scripts/extract_teacher_features.py` | Original pretrained Qwen (no_grad) | None | Collect H32 and eight layers' pre-RoPE K/V | Authored, 27B H100 unverified |
| 1: Offline warmup | `scripts/train_feature_projector.py` | Bridge only | Shared K/V Projector and KNorm | Normalized feature MSE | Implemented, H100 not needed once features saved |
| 2: CE adaptation | `scripts/train_27b.py` | One frozen pretrained Qwen + bridge | **Only bridge** | Causal next-token CE | Authored, 27B H100 unverified |
| 3: Teacher KL + Q/O LoRA | No qualified code yet | TBD | Small additional adapters | CE + optional KL | **Planned** |

Stage 1 controls: `--init source36` copies source pretrained K/V linear matrices. `--init random_linear` changes **only** those matrices, not copied KNorm. Stage 2 uses the matching `--projector-init` argument and rejects a warmup checkpoint whose recorded initialization contradicts it. The checkpoint stores the initialization and stage when produced by the updated scripts; older records without metadata are labelled legacy/unknown.

### Exact first training path (on actual H100)

~~~bash
python scripts/extract_teacher_features.py --revision "$MODEL_REVISION" --data "$TRAIN_JSONL" \
  --seq-len 128 --max-examples 16 --out artifacts/teacher_features
python scripts/train_feature_projector.py --features artifacts/teacher_features \
  --init source36 --out artifacts/warmup --steps 100 --batch-positions 64
python scripts/train_27b.py --revision "$MODEL_REVISION" --train-data "$TRAIN_JSONL" \
  --val-data "$VAL_JSONL" --projector-init source36 \
  --warmup-checkpoint artifacts/warmup/adapter.pt --seq-len 128 \
  --steps 16 --lr 2e-5 --out artifacts/train_b1
~~~

For random initialization control, change both init flags to `random_linear`, use separate warmup/train output directories, keep *all else* (data, seeds, tokens, optimizer, run time, checkpoint verification) comparable. **Do not silently relabel an old checkpoint** as one of the arms if its origin is unknown.

## H100 memory and scientific safeguards

- The 27.8B frozen BF16 backbone alone takes about 51.8 GiB before activations/caches/optimizer. Frozen *weights* do not eliminate backward activation memory when gradients propagate into the bridge through late layers. Start seq=128, batch=1, measure peak; do not promise one-H100 training fits.
- Source36 copy, offline feature warmup, CE are separate contributions. Record source model/dataset revision, teacher feature manifest, initialization flag, steps, train tokens, seed, wall and GPU time, actual HBM, held-out CE, full vs incremental/cache parity and saved adapter checksum. No GPU runs have been performed in this publication.
- B0 unmodified pretrained Qwen is an engineering anchor, **not** a fair trained control for B1b. A budget/data-matched *trained B0T* is still necessary and **not implemented**. Extra feature extraction and warmup cost must be included in full method cost.
- If CE improves but long-history and native benchmark scores collapse, reject a broad quality-preservation claim. If the shared cache reduces GQA KV but not TTFT, report it as storage improvement, **not** full CED speedup.
- Keep model weights/data, adapter checkpoints and raw logs on the authorized GPU host; GitHub is direct read/push evidence storage only. No Actions, workflow runs, or `.github/workflows`.

Sources: [Qwen3.8 weights](https://huggingface.co/Qwen/Qwen3.8-27B), [Qwen Transformers code](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py), [DeepSeek V4.1 technical report](https://arxiv.org/html/2609.19969).
