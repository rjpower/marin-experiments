# Async Pipeline Parallelism — Results

## STATUS: COMPLETE — NEGATIVE RESULT (measured)

Async no-flush pipeline parallelism (P=8 stages, 1 H100/stage, experts-local,
params-resident, per-stage Muon) was built, verified for schedule correctness, and
**measured on a real 8×H100 node**. It does **not** beat the EP+FSDP baseline — it is
~5.7× slower. The bottleneck is structural (single-threaded eager JAX dispatch cannot
overlap 8 device streams), not a tunable. Recommendation: **do not pursue PP** for this
model/hardware; keep the EP+FSDP pretrain (`run2-e128k8`, 187K tok/s, 15.1% MFU).

All numbers below are MEASURED (`[PP_THRUPUT]`/`[PP_RESULTS]` log lines), at the exact
production geometry (16L × 1536D × 128E/8K, B=16, S=4096, synthetic data SP_SYNTH_DATA=1).

## MEASURED RESULTS

| Run | Schedule / transport | tok/s (steady median) | MFU | step_ms | vs EP+FSDP |
|-----|----------------------|-----------------------|-----|---------|------------|
| EP+FSDP baseline (`run2-e128k8`) | sharded single-jit over full mesh | **187,000** | **15.1%** | — | 1.0× |
| pp6 | forward-sweep + sequential backward-sweep | 21,500 | 0.52% | ~3050 | 0.115× |
| pp7 | 1F1B, transport interleaved between dispatches | 21,500 | 0.52% | ~3050 | 0.115× |
| pp8 | 1F1B, **all transports deferred** off dispatch path | **32,702** | **0.79%** | 2004 | **0.175×** |

pp8 steady state (steps 16→80) is flat and stable: 31.9K–32.9K tok/s, 0.78–0.79% MFU,
~2.0 s/step. (Step-8 chunk = 2.5K tok/s is the one-time pipeline-fill + first-touch
allocation; excluded from the median, as designed.) Loss is flat ~11.77 — expected, this
is an 80-step throughput benchmark on random synthetic tokens, not a convergence run.

### What each fix bought (measured deltas)
- **1F1B vs sweep** (pp7 vs pp6): **0×** — identical 21.5K. The schedule shape is
  irrelevant when the stages don't actually overlap.
- **Deferring all cross-device transfers** off the dispatch thread (pp8 vs pp7):
  **1.52×** (21.5K → 32.7K). Real, but the transport was only ~1/3 of the per-step
  time; removing it from the critical path does not create cross-device overlap.

## Root Cause (the negative result, explained)

A ~2.0 s step with 8 stages is **~0.25 s/stage running serially**. Full overlap would put
the step time at ~one stage time (~0.25 s → ~256K tok/s). We never get there.

The pipeline is implemented as **eager, single-Python-thread, per-device `jit` dispatch**:
each tick the host loops `for s in range(8): fns[s].forward(...)` then the backwards, one
device per call. Two things keep this serial regardless of schedule:

1. **Cross-device `device_put` blocks the dispatch thread.** A device→device transfer that
   routes through host does a synchronous `device_get` (waits for the source to be ready)
   on the only thread that issues compute. pp7 paid this between every stage. pp8 defers
   all transfers to after all dispatches — which is why pp8 is 1.5× faster — but it is not
   enough to overlap.
2. **Single-thread eager dispatch cannot saturate 8 device streams.** Even with transfers
   removed from the critical path, issuing 8 independent `jit` calls from one Python thread
   does not get the 8 GPUs computing concurrently: per-call host dispatch latency plus the
   per-tick weight-update dependency chain (each stage updates its own weights every tick,
   so tick t+1's stage-s forward waits on tick t's stage-s optimizer) serialize the work.

This is exactly why EP+FSDP wins: it is **one** `jit`'d computation over the full 8-GPU
mesh, so XLA schedules all-device compute and collectives together. Our PP is 8 separate
eager computations the host feeds one at a time.

`grug-moe-pp/thread_probe.py` PASSES — it proves a *background transport thread* can move
arrays concurrently with main-thread dispatch without deadlock. That is the mechanism the
real fix would need, but it is a different (threaded/multi-process) execution model than the
eager loop measured here.

## Did experts-local free HBM for chonkier experts?

Partially confirmed, but moot. With EP=1 per stage each device holds only 2 of 16 layers'
worth of all 128 experts, fully resident, no FSDP all-gather — pp8 ran the full E128
production geometry on 8×H100 with **no OOM** (the EP+FSDP baseline needs `save_moe` remat
and sits near the fragmentation edge above ~E144). So PP does free per-device memory. But
freed HBM is worthless when the pipeline can't keep the GPUs busy: at 0.79% MFU you cannot
exploit bigger experts. No memory-headroom sweep was run — there is no point chasing
capacity on a 6×-slower trainer.

## Staleness token-tax

Not measured as a loss-vs-tokens curve: it would only matter if PP were throughput-
competitive, and it is not (6× slower kills it before staleness is even relevant). The
schedule's staleness is, however, verified exactly on CPU (`smoke_async_pp.py`,
`test_staleness_profile` PASS): 1F1B grad delay = 2·(P−1)−s ticks, i.e. stage 0 applies
gradients 14 ticks stale, the last stage fresh. At P=8 that is a *large* delay; the CPU
toy (linear model, no Muon, high LR) diverges under it, a reminder that P=8 async-no-flush
staleness is aggressive — another reason not to pursue this path.

## Go / No-Go

**NO-GO.** Measured 32.7K tok/s / 0.79% MFU is 5.7× slower than EP+FSDP's 187K / 15.1%.
The GO bar (≥250K tok/s or ≥20% MFU) is missed by an order of magnitude, and the gap is
structural, not parametric. Keep the EP+FSDP pretrain.

### If PP were ever revisited (not recommended for this model)
The only path to real overlap is to abandon single-thread eager dispatch:
- **multi-process JAX**, one process per GPU, explicit send/recv (`ppermute`) collectives
  between neighbor stages, each process driving its own device stream; or
- a **single `shard_map`/manual-collective pipeline** `jit`'d over the whole mesh (let XLA
  schedule the stages), which is essentially what EP+FSDP already does.
Both are substantial rewrites with uncertain payoff given EP+FSDP already runs at 15.1% MFU.

## Schedule-correctness verification (passed before the GPU runs)

- `smoke_async_pp.py` (CPU, 8 simulated devices): **4/4 PASS** — warmup fill, buffer depths
  exact (`act_fifo[s]` depth 2·(P−1−s), `label_pipe` depth P−1), staleness profile exact
  (`[14,13,12,11,10,9,8,7]`). The schedule is correct; the execution model is the problem.
- Attention/CE memory: the original 8-GPU OOM was the **attention score matrix**
  `f32[256,4096,4096]` at stage 0 (reference attention in f32), NOT the logits. Fixed with
  `gpu_fa4_cute` + bf16 compute + packed `segment_ids` + heuristic geometry; pp6/pp7/pp8 all
  ran clean. (The relayed "full-logits CE" diagnosis was incorrect for this failure; the CE
  head already used the chunked `fused_linear_softmax_cross_entropy_loss`.)

## Files

| File | Description |
|------|-------------|
| `async_pipeline.py` | Core: 1F1B schedule, per-stage buffers, per-stage Muon, bf16 cast, segment_ids, **deferred transport** |
| `smoke_async_pp.py` | CPU schedule-correctness tests (PASS 4/4) |
| `train_pp.py` | H100 entry point + `gpu_smoke()` (SP_PP_SMOKE=1, 1-GPU) |
| `iris_jobs.py` | `pp_smoke` (1-GPU verify) + `pp_async` (8-GPU benchmark) sweeps |
| `launch.py` | SP_PP_MODE=async dispatch hook |
| `grug-moe-pp/thread_probe.py` | Proves the threaded-transport mechanism (the real-fix prerequisite) |
| `h100_pp_model.py` | Parametric model — PREDICTIONS ONLY, superseded by the measurements above |
