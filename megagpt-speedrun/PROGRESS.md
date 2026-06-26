# megagpt-speedrun — PROGRESS

Live working notes. Goal: **best perplexity in 24h wall-clock on 8×H100** (CoreWeave
`cw-us-east-02a`), holding out ~3–4h for an SFT cooldown to get a chat-capable model.
"NanoGPT speedrun, scaled up." Two architectural bets (from the user):
1. **Reduce the CE/embedding dim** (factorized/ALBERT embedding, `d_e=512`), keep vocab 128256 —
   move compute out of the LM head into the experts. **Do NOT retokenize.**
2. **Go bigger + deeply sparse**: largest MoE that fits, large `E`, low active fraction.

---

## Architecture (grug MoE)
- `D=1536`, `d_e=512` (factorized embedding), `E=256` experts, `K=8` top-k, `seq=4096`,
  `layers=16`, `I_expert=768`, vocab 128256. ≈ **15B total / 0.75B active**.
- DeepSeek-V3 loss-free bias balancing, QB-β routing, GatedNorm, Exclusive Self-Attn (XSA),
  sigmoid combine, router z-loss, SwiGLU experts, sliding-window attn, RoPE, QK-norm, AdamH.
- Mesh: explicit JAX, EP=8 / TP=1 / replica=1 (EP shards experts AND batch).
- `RAGGED_DOT_IMPL=triton` is REQUIRED on GPU (the "auto" path OOMs at this geometry).
- Attn = `gpu_fa4_cute` (FA4). `FAST_QB`/`approx_max_k` is TPU-only — ~no-op on GPU.

## Infra
- Submit: `KUBECONFIG=~/.kube/coreweave-iris-gpu uv run iris --cluster=cw-us-east-02a job ...`
  (`job list`, not `ls`; kill is prefix-match → use full `/power/<name>`; UTC timestamps).
- Launcher: `./launch_cw.sh <name> <SP_TOKENS> <SP_STEPS|""> <GROUP> [extra -e ...]`
  (bakes the geometry; `LEVANTER_TS_CACHE_LIMIT=34359738368` + deep prefetch baked in).
- Data today: **R2** nemotron `tokenized/` TreeCaches (`SP_DATA=datakit` →
  `build_nemotron_datakit_mix`, 7 components). R2 creds are auto-injected on workers.
- SFT cooldown is wired: `SP_DATA=sft` (`build_sft_mix`: tulu3 0.5 + smoltalk 0.4 +
  OpenThoughts-Agent-v1-SFT 0.1, chat format, assistant-only loss) + `SP_INIT_FROM=<ckpt>` +
  `SP_SCHEDULE=linear SP_MIN_LR=0`.

## Key findings
- **MFU ~5%**, overhead/small-batch bound: per-expert matmul `M ≈ batch·seq/(E·shards) ≈ 128`
  ≪ H100's ~512 sweet spot. Attention/lm_head/router-sort dominate; model is NOT expert-bound
  (E64 only ~1.33× faster than E256). **Batch is the main throughput lever.**
- **E256 has better loss-per-token** (5.18 vs ~5.5 @66M tok) — the deep-sparsity bet is sound
  *per token*; it's just slower per second and was hurt by data stalls. The architecture was
  never the bottleneck — the data loader was.
- **R2 data loader**: the silent multi-hour hangs were an undersized TensorStore read cache
  (64 seqs/1MB chunk; default 1GB pool < block-shuffle working set → R2 re-fetch thrash). Fix =
  `LEVANTER_TS_CACHE_LIMIT=32GB` (run touches only ~21–32GB unique). With the fix, a **solo**
  run is clean (spr1: 1.6 it/s, 0 stalls). **4× concurrent** R2 cold-read streams still thrash
  → that's what cwobject fixes.

## cwobject (CoreWeave cluster-local object storage) — UNLOCKED
- Endpoint `https://cwobject.com`, virtual-hosted only (rejects path-style). Buckets:
  `marin-us-east-02a` (region `US-EAST-02A`), `marin-us-west-04a`. Creds = `CW_KEY_ID`/`CW_KEY_SECRET`.
- `datakit/store_8ac06c74/` = a complete llama3-tokenized corpus BUT in **raw datakit format**
  (100,810 separate per-part mini-stores) — **not** a loadable levanter TreeCache. The loadable
  `tokenized/` TreeCaches live only on R2 → **we mirror R2 caches to cwobject** (see below).
- **tensorstore needs 0.1.84** (PR google/tensorstore#285, merged 2026-02-27). 0.1.81 (shipped)
  rejects the config. **WORKING READ RECIPE** (verified — decodes coherent llama3 text):
  ```json
  {"driver":"s3","bucket":"","path":"<key>",
   "endpoint":"https://marin-us-east-02a.cwobject.com",
   "aws_region":"us-east-1","aws_credentials":{"type":"environment"}}
  ```
  with `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` = CW creds. Empty `bucket` + bucket-subdomain
  endpoint = virtual-hosted; no proxy, no `host_header`. (s3fs/boto already work via
  `addressing_style=virtual`; only tensorstore needed the version bump.)
- cwobject **write** verified from outside the cluster (s3fs virtual). R2 read creds are
  worker-only → the mirror must run as a worker job.

## Experiments (constant-LR, b16, 5.4B-token horizon — comparable loss@wallclock)
| run | geometry | status | notes |
|-----|----------|--------|-------|
| spr1 | E64/b16 | RUNNING clean | 1.6 it/s, loss 3.43 @ ~8k/82.4k, 0 stalls — fast baseline |
| spr6 | E256/b16 | launched (compiling, no OOM → b16 fits) | tests deep-sparsity loss/tok vs E64 throughput |
| killed | spr2 E256/b8, spr4 E128/b16, spr5 E64/b32 | killed | R2 contention hangs (4× concurrent) |

Loss-vs-tokens so far: spr1 — 49M→6.12, 95M→4.47, 144M→4.41, 425M→3.73. spr2(E256/b8) —
34M→6.84, 66M→5.18 (better/tok, slower/sec, then hung on data).

## Plan
1. **Mirror** R2 `tokenized/` caches → cwobject (smart protocol: parallel, resumable,
   smallest-first so experiments bootstrap on the first completed cache). 4 concurrent
   experiment slots once cwobject-backed.
2. **Bump tensorstore → 0.1.84**; add a `cw` data mode (`build_*` pointing components at the
   mirrored cwobject caches via the recipe above).
3. **Settle config** (spr1 E64 vs spr6 E256) → pick the best loss@wallclock for 24h.
4. **24h production run** (WSD: warmup+stable, then decay) → **SFT cooldown** (LR→0 on chat
   data) for the interesting model.
5. **PRs**: open a marin PR (cwobject/virtual-host data-path support) via sub-agent; open the
   megagpt-speedrun PR (don't merge); `weaver issue close 310` when done.

## Status update (2026-06-26 ~17:10Z) — PRODUCTION LAUNCHED + cwobject train-path FIXED
- **24h PRODUCTION pretrain launched**: `megagpt-prod-pre` — E64/b16, **R2 full nemotron**
  (`SP_DATA=datakit`), **cosine-WSD** (`SP_WARMUP=0.02 SP_MIN_LR=0.0`, decay-to-0 over the run),
  `SP_TOKENS=7.0e9` ⇒ 106,812 steps ≈ 18.5h clean @105k tok/s. Perplexity/bpb headline taken at
  end. Then **SFT cooldown** (`SP_DATA=sft SP_INIT_FROM=<ckpt> SP_SCHEDULE=linear SP_REWARMUP
  ~0.03 SP_MIN_LR=0` ~3.5h) → chat model. R2 path proven by spr1 (same geom, 1.6 it/s clean).
- **cwobject training path FIXED + root-caused.** The leak: iris seeds `fsspec.config.conf['s3']`
  (from `FSSPEC_S3_*` env) with the **R2 endpoint**; fsspec injects that into every
  `S3FileSystem` as EXPLICIT kwargs, beating our `AWS_ENDPOINT_URL` override → executor
  `write_infos` PutObject hit **R2** with the 16-char CW key → `InvalidArgument: access key length
  16, should be 32`. (Two earlier dead-ends: `AWS_ENDPOINT_URL_S3` — iris doesn't set it; s3fs
  `client_kwargs['endpoint_url']` — collides with `init_kwargs` → "multiple values".) Fix in
  `cw_patch.py`: (a) import cw_patch FIRST in launch.py; (b) `_override_fsspec_conf()` rewrites
  `fsspec.config.conf['s3']` → cwobject; (c) `_patch_s3fs_client()` FORCEs top-level
  endpoint_url/key/secret + virtual addressing + us-east-1 and pops any client_kwargs endpoint_url;
  (d) region forced us-east-1 (iris sets it "auto"). Verified locally under the exact worker env
  (FSSPEC_S3_ENDPOINT_URL=R2 injected) AND on the worker: `cwA5` passes `write_infos` with 0 S3
  errors. Canary confirming first training step; then E64/E256/E128 sweep on cwobject proofpile.

## Status update (2026-06-26 ~16:10Z)
- cwobject is **working end-to-end**: tensorstore 0.1.84 (override marin-levanter's <0.1.82 cap)
  + `cw_patch.py` (empty-bucket virtual-host tensorstore spec; s3fs forced to virtual via an
  AWS config file; tolerant `mkdirs`). Verified: `TreeStore.open` loads the mirrored proofpile
  cache from cwobject (11.3M rows) and decodes correct llama3 tokens.
- Mirror (`megagpt-cwmirror`): smallest-first, ~220 MB/s. proofpile_2 (173 GB) DONE;
  starcoderdata (691 GB) copying; `MIRROR_MAX_GB=600` stops after those two (~865 GB bootstrap).
  Giant nemotron tiers (1.4–3.6 TB each) deferred.
- marin PR **#6686** open: `build_kvstore_spec` virtual-hosted support (env-gated, backwards-compat).
- Config sweep on cwobject proofpile (same-data, constant LR, 5.4B horizon): `cwA`=E64/b16,
  `cwB`=E256/b8 launched; will add cwC=E128/b16, cwD=E64/b32. spr1 (R2 nemotron E64/b16) kept
  as the nemotron reference (loss 3.35 @ 753M tok).
- Decision pending: E64/b16 (fast, fits, ~105k tok/s) vs E256/b8 (better loss/token ~6%, ~82k
  tok/s; E256/b16 OOMs). cwobject sweep settles it on clean concurrent wall-clock.
- Data plan: experiments on cwobject (fast, concurrent); **24h production on R2 full nemotron
  mix** (proven solo) unless a nemotron tier is mirrored in time.

## Ops gotchas
- Monitor for SILENT data-loader hangs (job stays "running", step frozen, "Data loading is
  taking a long time" warnings; `--max-retries` does NOT recover). See `ops.md`.
- Parser: step lines can read `2.16kit` — expand `k` before comparing.
- iris `job logs` mixes a prior killed attempt's tail; trust the latest `[arm]` line.
