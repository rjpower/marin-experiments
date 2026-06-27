# Async Pipeline Parallelism — Results

## STATUS: NO MEASURED THROUGHPUT YET

**There are currently zero measured `[PP_THRUPUT]` readings.** The first batch of
8-GPU jobs all OOM'd during compilation (root cause below, now fixed). Everything
in the "Predictions" section is a PARAMETRIC MODEL, not a measurement. Do not cite
it as a result. This file will only report measured numbers once a job emits real
`[PP_THRUPUT]` lines.

## Root Cause of the First Failure (diagnosed from real logs)

All three `pp2-*` jobs died at compile with `RESOURCE_EXHAUSTED: Out of memory
trying to allocate 16.00GiB`. The actual allocation site (from the job log):

```
jit(forward)/checkpoint/Block/CausalSelfAttention/bqhd,bkhd->bhqk/dot_general
%gemm_fusion_dot.27 = f32[256,4096,4096]
```

That is the **attention score matrix** `[B*H, S, S]` in **f32**, NOT the logits:
`256 × 4096 × 4096 × 4 bytes = 16 GiB`. The OOM was at **stage 0** (attention),
not the last stage. Three compounding bugs in the PP path, all now fixed:

1. **`attention_implementation` defaulted to `None`** → `reference_attention`,
   which materializes the full `[B*H,S,S]` score matrix. The EP+FSDP baseline sets
   `SP_ATTN=gpu_fa4_cute` (FlashAttention-4, never materializes the matrix).
   FIX: build the model config via the same `MoeAdamHHeuristic` + override path as
   `launch.py`, setting `attention_implementation="gpu_fa4_cute"`.

2. **f32 compute** (no mixed precision). Baseline runs `compute=bfloat16`; the FA4
   kernel ONLY accepts bf16/fp16 inputs (train.py:636-639). FIX: store f32 master
   weights per stage, cast to bf16 inside each stage's forward (`_cast_compute`);
   the vjp upcasts grads back to f32 for the optimizer/Muon.

3. **No packed `segment_ids`** on the attention mask. `gpu_fa4_cute` raises
   `NotImplementedError` when `mask.segment_ids is None`. FIX: `_build_stage_masks`
   now attaches segment_ids matching the synthetic-data loader (~1024-token docs).

Bonus geometry fix: the hardcoded `num_heads=16` gave the wrong `B*H`; the
heuristic-built config uses the production 12 heads / 3 KV heads / I=768 / 16 layers,
so the PP benchmark now runs at the EXACT same geometry as the run2-e128k8 anchor.

Note: the relayed diagnosis (full-logits CE materialization) was incorrect for this
failure — the logs show it was the attention score matrix at stage 0. The CE head
was already using the chunked `fused_linear_softmax_cross_entropy_loss`.

## Verification Plan (economical: 1-GPU before 8-GPU)

1. **CPU schedule smoke** (`smoke_async_pp.py`): PASS 4/4 — schedule correctness,
   staleness profile (all 8 stages fire at tick P-1=7), buffer depths exact.
   (Linear einsum model; does not exercise FA4/Pallas.)
2. **1-GPU stage smoke** (`iris_jobs.py pp_smoke --gpus H100x1`, SP_PP_SMOKE=1):
   builds production-geometry model, runs fwd+bwd for first/mid/last stage types on
   ONE device at real per-stage shape. Confirms the attention fix fits per-device
   memory WITHOUT burning an 8-GPU node. STATUS: <pending>
3. **8-GPU throughput** (`iris_jobs.py pp_async --gpus H100x8`): only after (2) passes.
   STATUS: <pending>

## Parametric PREDICTIONS (NOT measurements — from h100_pp_model.py)

Profiled EP+FSDP step decomposition (E128/K8/b16): compute 31% | ep_a2a 29% |
fsdp_comm 15% | opt 10% | bubble 15%. PP eliminates ep_a2a+fsdp_comm = 44%.

| Quantity | Predicted (unverified) |
|----------|------------------------|
| Raw PP speedup | ~1.9× |
| Staleness tax (per-stage τ, 15k steps) | ~1.16× |
| Net speedup | ~1.64× |
| tok/s at E128/K8/b16 | ~307K (baseline 187K) |
| MFU | ~24.8% (baseline 15.1%) |

These are hypotheses to be confirmed or refuted by measurement.

## MEASURED RESULTS

_(empty — no `[PP_THRUPUT]` reading yet)_

### Primary: E128/K8/b16 + Muon
```
tok/s:    <pending>     (baseline 187K)
MFU:      <pending>     (baseline 15.1%)
step_ms:  <pending>
avg_loss: <pending>
```

## Go/No-Go Criteria (to be evaluated against MEASURED numbers)

- GO if measured tok/s ≥ 250K (≥1.34× over 187K) OR MFU ≥ 20%.
- NO-GO (valid negative result) if PP cannot beat EP+FSDP after the fix — report
  the measured numbers and the bottleneck (P2P, sequential backward, opt overhead).

## Deliverables Still Outstanding

- [ ] A real `[PP_THRUPUT]` reading (tok/s + MFU) at E128/K8.
- [ ] Staleness token-tax: loss-vs-tokens of async-PP vs the non-pipelined baseline.
- [ ] Whether experts-local frees HBM for chonkier experts (memory headroom probe).

## Files

| File | Description |
|------|-------------|
| `async_pipeline.py` | Core: schedule, buffers, per-stage Muon, bf16 cast, segment_ids |
| `smoke_async_pp.py` | CPU schedule correctness tests (PASS 4/4) |
| `train_pp.py` | H100 entry point + `gpu_smoke()` (SP_PP_SMOKE=1, 1-GPU) |
| `iris_jobs.py` | `pp_smoke` (1-GPU verify) + `pp_async` (8-GPU benchmark) sweeps |
| `launch.py` | SP_PP_MODE=async dispatch hook |
| `h100_pp_model.py` | Parametric model (predictions only) |
