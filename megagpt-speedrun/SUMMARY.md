# megagpt-speedrun — state of the project

"NanoGPT speedrun scaled up": the best LM we can train in a **fixed ~24h on one 8×H100 node**
(CoreWeave `cw-us-east-02a`), holding the last few hours for an SFT cooldown. Two design bets:
(1) **factorize the CE/embedding dim** (keep vocab 128256, narrow d_e) to move FLOP out of the LM
head into the experts; (2) **go big + deeply sparse** (MoE).

## Result so far (run #1)

- **Model:** grug-MoE, **D=1536, factorized d_e=512, E=64, top-K=8, I_expert=768, 16 layers,
  seq=4096, vocab=128256, EP=8.** **3.97B total / 0.80B active (~20% active).**
- **Pretrain:** 5.4B tokens, 82,397 steps, ~15.2h, constant-LR (WSD stable phase). Job
  `megagpt-spr1-e64b16`. Final ckpt `s3://marin-na/marin/grug/sparsity/sparsity-fixed-d1536-E64-k8-de512-s0-st82397-8f6629/checkpoints/step-82397`.
- **Held-out bpb = 0.947** (frozen-weights eval on a held-out nemotron slice; per-component:
  starcoderdata 0.74, proofpile_2 0.94, nemotron web splits ~1.08).
- **SFT cooldown** (tulu-3 + smoltalk + OpenThoughts-Agent, linear LR→0): `cool2-sft`.

## Key learnings (what actually moves the needle)

1. **The run was throughput-limited by the CE kernel, not the model.** The production run used the
   OLD `pallas_gpu` fused-CE (~98K tok/s ≈ **~6% MFU**). The current `batched_xla` streaming CE (with
   the autotune-cache fix) does **192K tok/s ≈ 15.5% MFU = ~2× faster** at the *same* model. **The next
   run gets ~2× the tokens for free just by running on the current code.**
2. **The run is overhead-bound, not compute-bound.** Device-time: EP-dispatch collectives 29%,
   optimizer/scatter 25%, attention 22%, dense matmul <9%, MoE expert compute ~0.2%, ~25% bubble.
   → the lever for MFU is reducing EP-dispatch comm and/or making the matmuls bigger (bigger active model).
3. **"Free capacity via E" doesn't help if experts starve.** A/B (`lossab`): E256 (14.8B total, K8)
   at the run's token budget never improved past its ~30-min loss (each of 256 experts saw only ~3% of
   tokens) while E64 kept descending. Rule: **keep tokens/expert healthy** — spr1's E64/K8 gives each
   expert K/E=12.5% of tokens; E256/K8 gives 3.1% (starves). Going bigger-E needs either more tokens or
   higher K (so K/E stays up) — "more experts with *less* sparsity".
4. **MFU rises with bigger experts.** I_expert 768→1536→3072 gave MFU 15.5%→16.9%→19.2% (bigger
   per-expert GEMM = more arithmetic intensity), but 4× expert-FLOP → fewer tokens. Wider experts also
   add *active* params (more learning/token), unlike E↑ which is free total capacity.
5. **HBM is NOT the limit on model size.** Static weights+AdamH optstate is only **~10-12 GB of 80 GB
   per device** for the 3.97B model (~12 B/param, sharded over 8). The binding constraint is the **MoE
   ring-dispatch forward activation transient** (~29 GiB @ b24), which caps the *batch* (~b16), not the
   parameter count. We can fit **far** bigger total/active models (tens of B) — see "next run" below.
6. **`tokens × active_params = MFU × peak_FLOP × wall_time / 6`** (peak ≈ 7.9 PFLOP/s bf16 on 8×H100).
   So "useful compute" (the quality-relevant product) is maximized by **MFU × wall-clock** — independent
   of how you split params/tokens. `tokens × total_params = (useful compute) / active_fraction`.
7. **Factorized head works:** the LM head is ~7% of FLOP (≈3× smaller than an unfactorized 2·D·V head),
   freeing FLOP for experts. Bets validated.
8. **Infra gotchas that cost real time** (see below): content-addressed cache skips, the s3:// XLA
   autotune blocker, the silent data-loading hang, version-drift in the SFT chat formatter.

## Experiments run

| sweep / job | what it tested | finding |
|---|---|---|
| `anchor` / w9 | E64/K8 EP8 baseline (new CE) | **192K tok/s, 15.5% MFU** (was "8%" on old CE) |
| `dp` (w9) | EP=1 data-parallel vs EP=8 | DP is *not* faster (EP8 ≥ EP1); all-to-all isn't the bottleneck |
| `mem`/`best` (w12/13) | bigger batch + remat + chonky experts | batch capped ~b16 (ring-dispatch transient); recompute_all doesn't unlock it; ragged_all_to_all dispatch is a loser (4.2% MFU) |
| MFU ladder | I_expert 768/1536/3072 | 15.5% / 16.9% / 19.2% MFU (wider experts = higher MFU, fewer tokens) |
| `lhs`/`lossab` w11/14/15 | XLA latency-hiding flags; **loss-vs-wallclock E64 vs E256 vs chonky** | flags: no win (~55% NCCL already overlapped). **A/B: E256 free-capacity LOSES (experts starve); thin-E64 vindicated at this budget** |
| profiling (`SP_PROFILE`) | xprof device-time breakdown | overhead-bound (collectives 29%, opt 25%, attn 22%, matmul <9%) |
| `spr1` | the 24h production run | E64/K8, 5.4B tok, bpb 0.947 |
| `eval` (ev3) | post-hoc held-out bpb | frozen-weights eval path; spr1 = 0.947 bpb |
| `cool` | SFT cooldown from spr1 ckpt | weights-only graft + linear LR→0; chat model |

## How to run things on iris (`cw-us-east-02a`, 8×H100)

All via **`iris_jobs.py`** (the Python submitter — uses the iris client API, no bash). Sweeps are a
dict in that file; each arm is a set of `SP_*` env overrides on top of `BASE`.

```bash
# launch a named sweep (always pass a fresh --tag; the marin executor is content-addressed and will
# SILENTLY SKIP an identical run_id as "already succeeded" without training -> set SP_TAG via --tag)
uv run python iris_jobs.py <sweep> --tag <uniq>            # e.g. anchor / lossab / mem / best
uv run python iris_jobs.py <sweep> --tag <uniq> --only <arm>
uv run python iris_jobs.py cool --tag coolN --init-from <pretrain /checkpoints dir>   # SFT cooldown
uv run python iris_jobs.py eval --tag evN  --init-from <ckpt /checkpoints dir>        # held-out bpb

# monitor (one persistent controller tunnel; parses the [THRUPUT] hook -> tok/s, MFU, tok/24h)
uv run python monitor.py <job-name-substr> --gpus 8 --warmup 200
uv run python monitor.py <substr> --logs <job> --grep RE         # dump+grep a job's logs

# profile analysis (xprof trace from R2 -> device op-time rollup)
uv run python analyze_profile.py <run_id_substr>
```

### Key `SP_*` knobs (read by `launch.py`)

| env | meaning |
|---|---|
| `SP_EXPERTS` / `SP_TOPK` | E (routed experts) / K (active per token) |
| `SP_BATCH` / `SP_SEQ` | per-step batch / sequence length |
| `SP_EMBED` | factorized CE dim d_e (512); `SP_INTERMEDIATE` overrides I_expert |
| `SP_EP` / `SP_TP` / `SP_REPLICA` | expert / tensor / replica mesh axes (EP=8 for the headline) |
| `SP_REMAT` | `save_moe` (default fast) or `recompute_all` |
| `SP_SCHEDULE`/`SP_WARMUP`/`SP_MIN_LR` | `constant` (pretrain stable) / fraction / 0 (cooldown decays peak→0) |
| `SP_DATA` | `datakit` (R2 nemotron train) / `datakit_eval` (held-out slice) / `sft` / `fineweb` |
| `SP_TOKENS` / `SP_STEPS` | LR-schedule magnitude / num train steps (decoupled: pass pretrain SP_TOKENS for LR, SP_STEPS for horizon) |
| `SP_INIT_FROM` | weights-only graft from a prior ckpt (cooldown/eval); step→0, fresh optimizer |
| `SP_EVAL`/`SP_EVAL_ONLY`/`SP_VAL_SEQS` | wire a held-out bpb eval and run it once on the loaded ckpt, then exit |
| `SP_PROFILE`/`SP_PROF_START`/`SP_PROF_STEPS` | xprof capture |
| `SP_SYNTH_DATA=1` | random tokens (pure-throughput smoke; no data-loader cold-start/hang) |
| `SP_TAG` | salt the run_id (REQUIRED on re-runs to dodge the content-addressed cache skip) |

### Gotchas (each cost real time)

- **Content-addressed skip:** identical run_id → "already succeeded", no training, no `[THRUPUT]`. Always `--tag`.
- **s3:// XLA autotune blocker:** `JAX_COMPILATION_CACHE_DIR=s3://…` makes the `batched_xla` CE's
  `__triton_gemm` autotune write to a non-existent s3 fs → FATAL. Fixed: `JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none`
  (in `BASE` + launch.py). **Any new run must carry this.**
- **Silent data-loading hang:** loader blocks ("Data loading is taking a long time: N seconds, Waiting
  for K items"), step frozen, state stays `running`, `--max-retries` does NOT recover. Hit both the
  pretrain and the SFT cooldown. **ROOT-CAUSED + FIXED (see `SFT.md`):** the SFT `cool` sweep was missing
  `LEVANTER_TS_CACHE_LIMIT` (default 1GB ⇒ R2 re-fetch thrash ⇒ a hung R2 GET with no timeout ⇒ the
  loader's `get_batch()` blocks forever). NOT the inline tokenization — all 3 caches consolidated ~20 min
  before the stall. Fix: (a) `iris_jobs.py BASE` now exports `LEVANTER_TS_CACHE_LIMIT=32GB`; (b)
  `data.build_sft_mix` reads the STATIC pre-built cache (`auto_build_caches=False`, no zephyr build
  sub-job) and by default LOCALIZES the (3.3GB) cache to the worker's local disk ⇒ zero R2 reads during
  training ⇒ a hung GET is structurally impossible. Validated by a 12k-step CPU loader soak (past the
  ~9.8k hang point). Recovery for any *other* hang: kill the FULL job name, confirm released, relaunch
  identical → resumes from the last 30-min temp checkpoint. NB job logs MIX the killed attempt's tail with
  the new run — diagnose by ADVANCING step numbers / fresh timestamps (a real hang *climbs*).
- **Version-drift:** the worker runs INSTALLED wheels, not the dev tree. After a marin bump, re-validate
  actual call paths (the SFT chat formatter's `chat_template_kwargs` flipped from a static dict to a
  column-name string between locks → `TypeError: unhashable type: 'dict'` in the cache build).
- **bpb eval data path:** datakit components are `source=None` hardcoded `/train` caches with no
  `/validation` split → use `build_nemotron_datakit_eval_mix` (`flat_cache=True` at `/train` +
  `num_validation_sequences` slices held-out from train). Eval forward must `mp.cast_to_compute` (FA4
  rejects fp32).

## Next run: target ~4× tokens×params (in progress)

`tokens × active_params = MFU × peak × wall_time / 6`. Budget we're leaving on the table vs spr1:
**~2× from the fast CE** (spr1 ran the slow CE at ~6% MFU) **× ~1.6× from using the full 24h** (spr1 used
only 15.2h) = **~3× more tokens at the same model, before any design change.** HBM has tens-of-B of
headroom. Design direction under test: a **bigger model** (more experts AND/or bigger active via higher
K / wider experts / larger D) sized so experts stay fed (tokens/expert ≳ spr1's) and MFU rises toward
compute-bound. See `iris_jobs.py` `scale` sweep + this dir's analysis scripts.
