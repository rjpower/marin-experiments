# Async Pipeline Parallelism Results

## Summary

Async no-flush pipeline parallelism (P=8 stages, 1 H100 per stage) eliminates
the two largest EP+FSDP overheads: ep_a2a (29%) and fsdp_comm (15%).
Per-stage Muon (Newton-Schulz orthogonalization local to each device) replaces
the cross-device Muon without any all-reduce on the ortho update.

**Baseline (EP+FSDP, run2-e128k8)**: ~187K tok/s, 15.1% MFU at E128/K8/b16.

## Architecture

```
Stage s (1 H100):
  params:  L/P = 2 transformer layers, all E=128 experts (no EP)
  forward: receives hidden from stage s-1 via NVLink P2P
  backward: receives cotangent from stage s+1, computes local grad
  optimizer: per-stage Muon (Newton-Schulz) + AdamW for norms/router
  staleness: stage s sees grad from (P-1-s) ticks ago
             stage 0: τ=7 (stalest), stage 7: τ=0 (fresh)
```

## Schedule Correctness (CPU Smoke Test)

**Test**: `smoke_async_pp.py --stages 8 --ticks 40 --staleness`
**Backend**: 8 fake CPU devices (JAX CPU, linear einsum model)

| Test | Result |
|------|--------|
| Warmup (ticks 0..P-2): loss=None | ✓ PASS |
| Post-warmup (tick P-1+): loss=numeric | ✓ PASS |
| All 33 post-warmup losses finite | ✓ PASS |
| All 8 stages' params updated | ✓ PASS |
| bwd_buf depths = P-1-s for all s | ✓ PASS (exact) |
| All 8 stages first fire at tick P-1=7 | ✓ PASS |
| last_label_buf = P-1=7 items at end | ✓ PASS |

**Staleness profile matches grug_stage_tau**: all stages fire simultaneously at
tick P-1, exactly as predicted by `delay_optim.grug_stage_tau(num_layers=8, num_stages=8)`.

## Parametric Model Predictions (from h100_pp_model.py)

Based on profiled EP+FSDP step decomposition (E128/K8/b16):
- compute: 31% | ep_a2a: 29% | fsdp_comm: 15% | opt: 10% | bubble: 15%

PP eliminates ep_a2a + fsdp_comm = 44% of step time.

| Configuration | Predicted |
|---------------|-----------|
| Raw PP speedup (vs EP+FSDP) | ~1.9× |
| Staleness tax (per-stage τ profile, 15k steps) | ~1.16× |
| Net PP speedup | ~1.64× |
| Expected tok/s at E128/K8/b16 | ~307K tok/s |
| Expected MFU | ~24.8% |

Note: actual results depend on P2P transfer overhead (~10% estimated),
optimizer parallelism, and JIT compilation overhead.

## H100 Benchmark Jobs (Submitted, Results Pending)

| Job ID | Config | Status |
|--------|--------|--------|
| /power/pp2-e128k8p8 | E128/K8/b16/Muon | pending |
| /power/pp2-e64k8p8 | E64/K8/b16/Muon | pending |
| /power/pp2-e128k8p8nomunon | E128/K8/b16/AdamW | pending |

Monitor: `uv run python monitor.py pp2 --gpus 8 --warmup 10 --watch 120`

## Results (TBD — fill when H100 jobs complete)

### Primary: E128/K8/b16 + Muon (/power/pp2-e128k8p8)

```
tok/s: TBD      (baseline: 187K)
MFU:   TBD%    (baseline: 15.1%)
step_ms: TBD   (baseline: ~440ms)
avg_loss: TBD
```

### Secondary: E64/K8/b16 + Muon (/power/pp2-e64k8p8)

```
tok/s: TBD
MFU:   TBD%
step_ms: TBD
```

### Muon vs AdamW isolation (/power/pp2-e128k8p8nomunon)

```
tok/s (no Muon): TBD
MFU (no Muon):   TBD%
Muon overhead:   TBD%
```

## Memory Profile (Expected)

With P=8 stages, each stage holds 2 layers × 128 experts × 768×1536×3 bytes
= ~2 GB in BF16 parameters per stage + optimizer states (~4 GB) + activations
during backward (~2 GB) = ~8 GB per stage.

All 8 H100s have 80 GB HBM each, so memory constraint is non-binding.
The production EP+FSDP baseline requires ~40 GB for model params alone
(before optimizer states), so PP's per-stage memory is dramatically lower.

## Staleness Token Tax (Expected, from delayed-gradient-pp/REPORT.md)

| τ_max | Measured (delay_optim pp6 profile) | With weight-prediction |
|-------|-------------------------------------|------------------------|
| P-1=7 | ~1.33× at 6k steps, ~1.16× at 15k | ~1.23× at 15k |

PP-async's staleness profile (all stages fire at P-1=7) matches the pp6
profile from delayed-gradient-pp exactly. The 1.16× token tax at 15k steps
means we need ~16% more tokens to match non-stale baseline quality.

Net throughput advantage even accounting for staleness:
  ~1.64× net speedup × 1/1.16× staleness = ~1.41× tokens/wall-time

## Go/No-Go Decision (Pending H100 Results)

Decision criteria:
- GO if actual tok/s ≥ 250K (1.34× over 187K baseline)
- GO if MFU ≥ 20% (vs 15.1% baseline)
- NO-GO if P2P overhead > 25% of step time (would negate the benefit)
- NO-GO if memory OOMs on any arm

## Files

| File | Description |
|------|-------------|
| `async_pipeline.py` | Core implementation: schedule, buffers, Muon |
| `smoke_async_pp.py` | CPU schedule correctness tests (PASS 4/4) |
| `train_pp.py` | H100 training entry point (SP_PP_MODE=async) |
| `iris_jobs.py` | `pp_async` sweep: 4 arms (e128k8p8, e64k8p8, nomunon, b32) |
| `launch.py` | SP_PP_MODE=async dispatch hook |
| `h100_pp_model.py` | Parametric throughput model (predictions above) |
