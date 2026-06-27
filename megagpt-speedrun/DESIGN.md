# megagpt-speedrun — design & understanding

**Goal.** A "NanoGPT speedrun, scaled up": given a *fixed* 24-hour wall-clock budget on
**one 8×H100 node** (CoreWeave `cw-us-east-02a`), train the best language model we can —
lowest perplexity / bits-per-byte — and spend the last few hours doing **SFT in the
cooldown** so we end with a genuinely *interesting* (chat/instruction-capable) model, not
just a low training loss.

Two architectural bets, from the user brief:

1. **Reduce the "CE dimension" — the *embedding* dimension, NOT the vocab — to move compute
   into the experts.** (Factorized / ALBERT-style embedding: keep vocab 128256, no
   retokenize; narrow the token-table + LM-head width `d_e`.)
2. **Go big and deeply sparse:** the largest Mixture-of-Experts we can fit on 8×H100,
   with a low active fraction.

This project is a fork of [`adaptive-sparsity`](../adaptive-sparsity) (the grug-MoE
sparsity study) and reuses its model, heuristic, trainer, and launch plumbing.

---

## 1. Why this is the right bet (inherited findings)

The `adaptive-sparsity` REPORT measured, on the same grug MoE, where the FLOPs actually
go at a thin geometry (D=512, vocab 128256, seq 4096):

| component | K=1 FLOP share |
|---|---|
| **lm_head (D×vocab)** | **61.4%** |
| attention | 27.3% |
| shared expert | 4.4% |
| router proj | 2.9% |
| **routed experts** | **2.2%** |
| gated-norms | 1.7% |

The headline: **the LM head ate 61% of compute and the experts only 2.2%.** The model was
"over-provisioned" and sparsity bought almost nothing because there was nothing to
reallocate — the expert FLOPs were a rounding error next to the head. The speedup ceiling
from sparsity was only 1.33×.

That report's own "what we'd try next" is essentially this project: a geometry where
**routed experts are most of the FLOPs** (large D / intermediate, **small CE head**).
The lm_head FLOP/token is `2·D·V`. The clean way to shrink it *without* touching the
tokenizer is a **factorized embedding**: put the token table and LM head at a narrow
`d_e < D` and project `d_e↔D` with small matrices. The head cost drops from `2·D·V` to
`2·d_e·V + 2·D·d_e`, i.e. by ≈ `D/d_e` for the dominant `V` term — **vocab stays 128256, no
retokenize, the existing tokenized data is reused as-is.** The other levers that move compute
into experts are **grow expert FLOP** (more layers `L`, higher `K`, larger `I_expert`) and
**grow E** (expert count) — which adds *total params at ~zero active FLOP*, the "go big and
sparse" lever. (Shrinking the *vocab* would also cut `2·D·V`, but it forces a retokenize and
discards the existing 2.84T-token nemotron caches — explicitly rejected by the brief.)

---

## 2. Architecture decisions

### 2.1 Reduce the CE dimension: factorized embedding (keep vocab 128256)

**Decision (corrected from an earlier vocab-cut draft):** keep the llama3 vocab **128256**
and instead **factorize the embedding** — narrow the token table + LM head to a CE dimension
`d_e < D`, with small learned up/down projections to the model dim `D`. Concretely
(`model.py:Transformer`):

- `token_embed[V, d_e]` → lookup gives `[B,S,d_e]` → **up-project** `embed_up[d_e, D]` →
  `[B,S,D]` into the body. (Embedding lookup itself is ~0 FLOP.)
- body runs at full `D`; at the head, **down-project** `head_down[D, d_e]` → `[B,S,d_e]` →
  `output_proj[d_e, V]` for the fused-CE loss.
- When `d_e == D` (default) the projections are `None` and behavior is identical to the
  original model (backward-compatible; validated on the 8-device sharding harness).

**Why this and not a vocab cut:**
- It attacks the same 61%-head cost (head FLOP `2·D·V → 2·d_e·V + 2·D·d_e`) **without** a
  tokenizer change, so the **existing 2.84T-token nemotron caches are reused verbatim** — the
  brief's hard constraint ("we have a fully tokenized nemotron, don't give it up").
- No text-compression penalty (a smaller vocab needs more tokens/doc); cross-entropy stays
  directly comparable across all arms (same vocab).
- ALBERT showed a factorized embedding decouples vocab width from hidden width with little
  quality loss; here the motivation is FLOP reallocation, not param count.

**Headline:** at `D=1536, d_e=512` the output head drops from ~22% to **~7% of forward FLOP**
(measured in `geometry_explore.py`), and that compute moves into the deeply-sparse experts.

- **Metric:** we still report **bits-per-byte (bpb)** as the primary, vocab-robust metric;
  token-CE is also valid here since the vocab is fixed across every arm.
- **Knob:** `SP_EMBED` (launch) → `embed_dim` (heuristic) → `GrugModelConfig.embed_dim`.

### 2.2 Deeply-sparse, big MoE geometry (to be finalized empirically)

The geometry knobs (`heuristic.py:build_model_config`, `model.py:GrugModelConfig`): the
heuristic derives `num_layers`, `num_heads` (=D/128), `num_kv_heads` (4:1 GQA), `I_expert`
(=D/2→128-mult), `shared=D` all from **D**; `E`, `K`, adaptive knobs are passed in. To go
"big + deeply sparse" we **parametrize** `num_layers` and `I_expert` (currently hidden
functions of D) and push **E up, K low, I_expert fine-grained** (the REPORT found
fine-grained experts are what make sparsity actually pay).

**Sizing target:** *largest total params that fit 8×H100 (640 GB) with FSDP + expert
parallelism, at an active-param/compute budget that stays compute-bound and processes tens of
billions of tokens in ~20 h.* Reference points: nanochat trains a ~1.9 B **dense** model to a
usable chat model in ~24 h on 8×H100; an MoE at matched *active* compute should reach more
total capacity. So the rough target is **~1–2 B active params, ~8–16 B total** via many
experts.

Memory back-of-envelope: AdamH state ≈ 12 bytes/param (fp32 params + 2 fp32 moments), sharded
across 8 GPUs. Reserving ~400 GB for params+opt leaves headroom for activations (remat on) →
up to ~30 B total params is *theoretically* in budget; we will target the lower, safer end and
confirm fit empirically.

**Headline geometry (chosen; `geometry_explore.py`).** `D=1536, d_e=512, E=256, K=8,
I_expert=768 (heuristic default), 16 layers, seq 4096, vocab 128256`, EP=8:

| metric | value |
|---|---|
| total params | **14.86 B** |
| active params / token | **0.746 B** (active fraction **3.1%**) |
| output-head FLOP share | **7.0%** (vs ~22% un-factorized at this D) |
| fwd FLOP / token | 1.90 GFLOP |
| HBM / GPU | ~22.3 G (AdamH 12 B/param ÷ 8) → ~58 G headroom for activations |

Rationale: "go big" (≈15 B total) + "deeply sparse" (3.1% active) + factorized head (7%).
**E=256 not 512** — the [[grug-moe-sparsity-perf-ceiling]] memory found the router top-k sort
dominates device time at very high E; E=256 keeps the sort tractable while still deeply sparse.
**I_expert left at the heuristic default (768)** so the AdamH LR / batch / step sizing stay
self-consistent (LR depends on D/tokens/batch, not on I). Comparison candidates and the
factorization head-share deltas are in `geometry_explore.py`. *Fit + throughput confirmed by
the H100 fit-smoke (Task #8) before the 4h run; geometry may shift if HBM/throughput demands.*

**Sharding (single 8×H100 node).** Expert weights are `P(expert, data, model)` and the batch
axis is `(replica_dcn, data, expert)`, so **EP=8** does double duty: it shards the 256 experts
8-way (32/device, full D×I each) *and* shards the batch 8-way via the expert axis. Non-expert
params (attention/head/norms, small) replicate. This is memory-feasible and the canonical MoE
layout; `SP_EP=8 SP_REPLICA=1 SP_TP=1`.

### 2.3 Optimizer / speedrun techniques to add

The base already has the cheap NanoGPT wins: **RoPE, QK-norm, SwiGLU experts, sliding-window
attention, output z-loss + router z-loss, DeepSeek loss-free-bias balancing, untied
embeddings**, and **AdamH** (a scale-invariant, Muon-*spirit* Adam — but **without**
Newton-Schulz orthogonalization). Ranked additions (per the speedrun review):

1. **Complete Muon: add Newton-Schulz orthogonalization to AdamH** — the orthogonalization
   *is* the part that delivers Muon's documented ~1.35–1.5× sample-efficiency win, and it was
   the single biggest lever in speedrun history. **MoE benefits more than dense** (Moonlight/
   Kimi trained a 16B-MoE entirely on Muon at ~52% of AdamW FLOPs). Must: apply NS **per
   expert** (batched/vmapped over the expert axis — never across the expert dim), add Moonlight
   RMS-matching shape-scaling `0.2·√max(d_in,d_out)` + weight decay for the rectangular expert
   matrices, and **keep Muon on through the SFT cooldown** (switching optimizers erases the
   gain). *Highest leverage; moderate effort; well de-risked.*
2. **Value embeddings** — extra learned tables injected into the attention value stream;
   params-without-FLOPs, orthogonal to the MoE FFN. *Strong second; moderate effort.*
3. **Batch ramp + per-group LR** — matters specifically for a *long* (24h) run where
   large-batch efficiency and throughput dominate; Muon tolerates larger batches than AdamW.
   Per-group LRs are also needed to make NS behave across heterogeneous tensors.
4. **FP8 — on the experts/MLP, not the head.** The head is now cheap (we shrank vocab), so the
   classic FP8-head win is small for us; FP8 on the expert up-proj is the current-WR move. *Defer.*
- **Skip:** logit soft-cap (redundant with our output z-loss); ReLU² experts (we have SwiGLU,
  no established win inside experts).

**Phasing:** the 4-hour run uses the base AdamH (proven) to validate the pipeline and get a
baseline. Newton-Schulz + value embeddings go in for the 24-hour run if they pass a smoke A/B.

### 2.4 LR schedule = WSD with an SFT cooldown tail

The trainer's schedule is WSD (`warmup` → stable → linear/cosine `decay`). Plan: warmup ~2-3%,
long stable phase at peak LR, then a **cooldown tail (~15-20% of steps)** that simultaneously
(a) anneals LR → 0 and (b) **ramps the data mixture from pretrain → SFT/high-quality** via
`lm_varying_mixture_data_config` (marin's idiomatic "pretrain/midtrain/SFT are all just data
phases" pattern, `experiments/references/reference_training_pipeline.py`). Total steps must be
fixed up front so the schedule lands correctly (the loop has **no wall-clock stop** — we
convert 24h→steps from measured throughput).

---

## 3. Cluster: CoreWeave `cw-us-east-02a` (8×H100)

From tunix/MARIN.md §11 (verified end-to-end) + exploration:

- **Submit:** `uv run iris --cluster=cw-us-east-02a job run --no-wait --gpu H100x8
  --enable-extra-resources --extra gpu --cpu 32 --memory 512GB --disk 200GB ...`.
  **No `--region`** (single-region k8s). `--gpu H100x8` = full node.
- **Local prereqs:** `export KUBECONFIG=~/.kube/coreweave-iris-gpu` (context
  `marin-gpu_US-EAST-02A`), `R2_ACCESS_KEY_ID`/`R2_SECRET_ACCESS_KEY` (present), and
  `marin-iris[controller]` in deps (added) so the CLI can reach the k8s controller. Prove with
  `iris --cluster=cw-us-east-02a cluster status` (k8s shows `Workers: 0/0` — normal; pods
  dispatch onto the static 32×H100 pool).
- **GPU deps:** minimal `gpu` extra = `jax[cuda13]==0.10.0 + nvidia-cublas + nvidia-nccl-cu13`
  (the full `marin-core[gpu]` is unsatisfiable — torch/cublas pin clash). Attention =
  `reference` for now (FA4 needs flash-attn-4 → torch → the conflict). `gpu`⊥`tpu` so the lock
  forks.
- **Mesh adapts automatically:** `compact_grug_mesh` reads `jax.devices()` → on one host with
  8 GPUs it's `(replica=1, data=8, expert=1, model=1)` = pure FSDP over `data`. For a model too
  big to replicate, set `expert_axis_size>1` (EP; needs `E % expert_axis_size == 0`) and/or a
  `model` axis. EP backend `ring` (default) or `deepep` (intranode GPU).
- **DATA: CoreWeave cannot read `gs://`** — it's wired to **R2 `s3://marin-na/…`**. The
  `gs://` nemotron paths are unusable, **but the same `tokenized/` levanter caches are mirrored
  to R2** at `s3://marin-na/marin/tokenized/<rel>/train/` (real TreeCaches with
  `shard_ledger.json`; vocab 128256). `data.build_nemotron_datakit_mix()` (default
  `SP_DATA=datakit`) reads these — **verified 2.84T tokens** across 7 comps. No retokenize.
  (Dead ends: the `datakit/tokenize` raw parquet and the `datakit/store_b9f9b109` store both
  lack ledgers. See §4.) Checkpoints go to `MARIN_PREFIX` (R2) — fine.

---

## 4. Data plan

**No retokenize** (vocab stays llama3 128256), so the existing R2-mirrored caches are used
directly.

- **Pretrain:** `s3://marin-na/marin/tokenized/` nemotron mixture
  (`data.build_nemotron_datakit_mix`, `SP_DATA=datakit`). 7 verified-loadable comps:
  nemotron_cc {hq_actual 537B, medium_high 489B, medium_low 839B, low_actual 384B, low_synth
  322B} + starcoderdata 216B (code) + proofpile_2 49B (math) = **2.84T tokens** — no epoching
  at the ~tens-of-B-token budget of a 24h run. Weights are a **quality tilt** (`hq_actual`
  up-weighted) because the two largest natural splits — `hq_synth` and `medium` — have **no
  ledger on R2 under any hash** and can't be loaded. **Cache-hash gotcha:** of the ~2 hashes
  per split on R2, only one carries a ledger (verified `medium_low-5b94a4`, `proofpile_2-5ba7ac`).
- **Dead ends (documented so we don't relitigate):** the marin `datakit/tokenize/<ds>/train/`
  raw parquet (`{id, input_ids}`) is *not* a levanter cache (no `shard_ledger.json` → `TreeCache.load`
  fails); the `datakit/store_b9f9b109` store mirror is also missing its ledgers; the
  `grug-moe-cw-may-*` R2 runs trained on **synthetic** tokens (perf benchmarks). Only the
  `tokenized/` caches load.
- **SFT cooldown:** small, high-yield sets, **already / re-tokenized to the same llama3 vocab**
  (no special-token surgery needed). Candidates: **`allenai/tulu-3-sft-mixture`** and/or
  **`HuggingFaceTB/smoltalk`** (format installs fast) + a capability spice —
  **`open-thoughts/OpenThoughts-Agent-v1-SFT`** (the user's named agentic set; reusable
  ChatML/masking code in `../openthoughts-agent`). A chat template with a `{% generation %}`
  block gives assistant-turn loss masking. **Availability of these on R2 (tokenized) is the
  open item for Task #10** — if absent, tokenize-to-R2 on the worker (a one-off, cheap vs the
  pretrain) or fall back to whatever chat SFT cache is already mirrored.
- **Caveat carried forward:** a few-hour cooldown *amplifies what pretraining seeded* — SFT
  reliably installs chat **format** fast, but **content competence is bounded by the ~20h
  pretrain** (see [[delphi-vs-qwen-chat-sft]]). Weight the cooldown toward format + a narrow
  capability we actually pretrained for.

---

## 5. Execution plan (phased to de-risk)

1. **Infra bring-up (Task #7 — DONE).** GPU lock + `uv sync --extra gpu`; local 8-CPU
   explicit-sharding harness (`_smoke_gpu_shard.py`); `cluster status`; **100M-scale H100 smoke
   trained 25 steps** (8-GPU mesh, R2 data load, checkpoint to R2). Fixed the GPU attention
   sharding bug (reference_attention head-axis vs TPU splash) — see memory.
2. **Factorized embedding + data + geometry (Task #8).** DONE: factorized embedding in
   `model.py`/`heuristic.py`/`launch.py` (`SP_EMBED`), validated on the local harness
   (backward-compat + `d_e<D` + TP/EP sharding-invariant); **R2 nemotron data path solved**
   (2.84T tok, §4); headline geometry chosen (`geometry_explore.py`, §2.2); mesh knobs
   (`SP_EP/SP_TP/SP_REPLICA`) wired through `train.py`. Remaining: confirm HBM-fit + measure
   sec/step→tokens/s on the H100 fit-smoke to size the step count; Newton-Schulz/value-embeddings
   are optional 24h-run levers (see §2.3), gated behind a smoke A/B.
3. **4-hour run (Task #9).** Target geometry, ~4h of pretrain, then **write an initial report
   as a GitHub issue** (bpb/perplexity vs tokens, throughput/MFU, fit, the design decisions)
   for user feedback before committing the full day.
4. **24-hour run (Task #10).** ~20h pretrain (WSD) + **~3-4h SFT cooldown** (LR anneal + data
   mixture ramp to SFT). Final report + PR.

---

## 6. Open questions / risks (flag for the 4h-report checkpoint)

- **R2 data pipeline:** SETTLED — `tokenized/` caches load on R2 (§4); the raw `datakit/tokenize`
  parquet and `store_b9f9b109` do not (no ledger).
- **HBM fit + throughput at 15B:** the open #1 — the fit-smoke confirms the model fits 8×80G and
  gives sec/step → tokens/s → the step count for 4h/24h. (Geometry may shrink if it doesn't fit
  or throughput is too low.)
- **`reference` attention perf** at seq 4096 on H100 may be slow/memory-heavy; mitigations:
  shorter seq (2048) and/or window warmup; resolve FA4/torch conflict if attention dominates.
- **Active-param floor:** too-deep sparsity (3.1% active) may raise the loss floor; the fit-smoke
  + 4h loss curve check that the model is genuinely compute-bound and not under-parameterized
  per token. E=256/K=8 chosen as a tractable-sort, not-too-sparse point.
- **LR heuristic at this scale:** the AdamH LR fit was measured at smaller models; the 15B/EP
  geometry is an extrapolation — sanity-check the loss curve on the 4h run. Factorized embedding
  keeps the vocab fixed, so the head-related LR assumptions are unchanged.
- **Newton-Schulz on stacked experts (if added):** must orthogonalize per-expert (batched), with
  RMS shape scaling + weight decay; risk of instability if mis-applied — gate behind a smoke A/B.

*Bottom line:* shrink the head, pour the freed compute into a large deeply-sparse expert pool,
add the one missing speedrun lever (Newton-Schulz/Muon), train ~20h on 8×H100, and finish with
an SFT cooldown. Judge by bits-per-byte, and end with a model that can actually hold a
conversation.
