# Next run: optimizing for 4× tokens×params (design exploration)

**Task:** push ~4× the `tokens × params` of run #1 (spr1) into the next 24h run, then compare final
quality. Explore the design space ("more experts, less sparsity"). User's prior: *"I don't believe
the biggest model you can fit in 320G of RAM is 3B."*

## TL;DR

- **The user is right that memory isn't the limit** — static weights+optimizer for the 3.97B spr1 is
  only ~6 GB of 80 GB/device; HBM *statically* holds ~26B params. spr1's 3.97B was a **throughput**
  choice, not a memory ceiling.
- **But the biggest model that TRAINS WELL in 24h is ~7.6B, not the biggest that FITS.** The binding
  constraint is the **MoE forward ring-dispatch activation transient** (∝ batch·seq·D, materializes all
  tokens on every device, *unfreeable by rematerialization*), which caps the full-throughput batch at
  **b16**. At b16 total params top out at **~E128/K8 (≈7.6B) reliably**, and the OOM is
  **nondeterministic near the wall** (E144 & E176 ran clean but E160 *between them* OOM'd at startup) — so
  E144-E176 is a fragmentation edge, not a safe ceiling. Bigger models only run at **b8 → fewer tokens
  AND lower MFU (11.7% vs 15.1%) AND starved experts** → *worse* quality.
- **Recommendation: `D1536 / d_e512 / E128 / K8 / I768 / 16L / seq4096 / EP8 / b16 / save_moe`** — the
  largest thin-expert model that runs **clean at full-throughput b16 with margin**. vs spr1 this is **1.9×
  bigger (total params)**, runs at spr1's full speed (≈187K tok/s, 15.1% MFU), keeps experts **fed**
  (0.84B ≥ spr1's 0.67B tok/expert), and delivers **≈4.8× tokens×total / ≈2.6× tokens×active** over spr1
  — clearing the 4× target on the size-weighted product. *(Aggressive alt: E144 = 8.5B, 5.3× tokens×total,
  still fed at 0.74B, but at the fragmentation edge — usable with `--max-retries` to re-roll a startup OOM.)*
- **The naive "max tokens×params" answer (E256/K8 @ b8 = 7.3× tokens×total) is a trap:** its experts
  starve (0.33B tok/expert) and its *useful* compute (tokens×active) is *lower* than the b16 options.
  The A/B already showed E256 doesn't convert to quality. **tokens×TOTAL via extreme sparsity ≠ quality.**

## The metric, made precise

`tokens × active_params = MFU × peak_FLOP × wall_time / 6`. So the quality-relevant "useful compute"
depends only on **MFU × wall-clock** — not on how you split params vs tokens. Two facts about run #1:

1. **spr1 ran the OLD slow CE at ~6% MFU.** The current `batched_xla` CE is ~2× faster (15.5% MFU).
2. **spr1 used only 15.2h** of the 24h budget.

So **the same model on today's code at the full budget is already ~2.6× the tokens (and ~2.6× both
products) — for free.** That free 2.6× is most of the way to 4× before any design change; the design
job is to spend the remaining headroom on a *bigger model that still trains well*.

`tokens × total_params = (useful compute) / active_fraction` — raising total via more experts (lower
active fraction) multiplies tokens×total, **but only converts to quality while experts stay fed**.

## Measured decision table (smoke sweeps sc1/sc2/sc3, new CE, 8×H100, 20h pretrain + 4h SFT)

```
config              tot(B) act(B)  b   tok/s  MFU%  tok(B)  xTOTAL xACTIVE t/exp  fit
spr1 (run#1 geom)    3.99   0.82  16   192K  15.5   13.8   2.59x  2.62x  1.73   CLEAN (CE+24h alone = 2.6x free)
E128/K8              7.62   0.82  16   187K  15.1   13.5   4.82x  2.56x  0.84   CLEAN  <- RECOMMEND
E144/K8 (edge)       8.52   0.82  16   185K  14.9   13.3   5.34x  2.53x  0.74   CLEAN but near edge  <- aggressive
E176/K8 (edge)      10.34   0.82  16   181K  14.6   13.0   6.33x  2.48x  0.59   CLEAN but E160 OOM'd; STARVE
E160/K8              9.43   0.82  16   OOM    -      -      -      -      -      OOM (yet E176 fit -> nondeterministic)
E64/I1536 chonky     7.62   1.27  16   148K  17.0   10.7   3.81x  3.14x  1.33   MARGINAL (OOM@step40); best xACTIVE
E256/K8 @b8         14.87   0.82   8   145K  11.7   10.4   7.29x  1.99x  0.33   B8 only, STARVE  <- trap
E192/K8            11.24   0.82  16   OOM    -      -      -      -      -      OOM
E128/I1024          7.62   0.82  16   OOM    -      -      -      -      -      OOM
E256/K8 @b16       14.87   0.82  16   OOM    -      -      -      -      -      OOM
E256/K16 @b16      14.87   1.28  16   OOM    -      -      -      -      -      OOM
D2048/E64           9.12   1.72  16   OOM    -      -      -      -      -      OOM (bigger D = bigger transient)
D2048/E128         17.58   1.72  16   OOM    -      -      -      -      -      OOM
```
(xTOTAL/xACTIVE = multiple of spr1's product. STARVE = tok/expert < spr1's healthy 0.67B.)

## What the smoke sweeps established

1. **b16 batch wall = MoE forward ring-dispatch transient, total-params-gated.** OOM allocations scale
   with *total params* (E256/14.9B → 23.5 GiB; D2048/17.6B → 31 GiB), and **`recompute_all` does NOT
   fix it** (e128i1536 OOM'd with recompute_all) — confirming it's the forward all-gather transient,
   not the saved-activation buffer. As static params grow, they eat the headroom the b16 transient needs.
2. **Thin experts (small I) are the memory-efficient capacity lever at b16.** E128/K8 (I768) is clean;
   the *same 7.6B total* as chonky E64/I1536 (I1536) but the chonky version is memory-marginal (2× the
   per-expert activation tips the transient over). So add capacity via **more thin experts**, not wider.
3. **Chonky experts give the most useful compute (3.14× tokens×active, MFU 17%) — if you can fit them.**
   Bigger per-expert GEMM raises arithmetic intensity. But chonky is memory-marginal at b16; it would
   need b8 (fewer tokens) or a memory unlock to run reliably for 20h.
4. **b8 is a bad trade:** the 14.9B model at b8 runs at 145K/11.7% — fewer tokens *and* lower MFU than
   the b16 thin models, and its experts starve. More total params, less of everything that matters.

## Recommended config + launch

`D1536 / d_e512 / E[CEILING] / K8 / I768 / 16 layers / seq4096 / EP8 / b16 / save_moe`, WSD: constant-LR
pretrain (~20h) then SFT cooldown (~4h, linear LR→0). Same harness as spr1; geometry differs only in E.

The `run2` sweep arm is ready in `iris_jobs.py` (E128/K8, real datakit, constant-LR WSD, ~13.5B tok /
~20h at 187K tok/s). To run the full comparison:

```bash
# 1. pretrain (~20h, constant-LR stable phase)
uv run python iris_jobs.py run2 --tag run2
# 2. held-out bpb on the final ckpt (compare directly to spr1's 0.947)
uv run python iris_jobs.py eval --tag run2eval --init-from <run2 final /checkpoints>
# 3. SFT cooldown (~4h, linear LR->0) -- see SFT caveat below
uv run python iris_jobs.py cool --tag run2cool --init-from <run2 final /checkpoints>
```
(For the aggressive 8.5B variant, set `SP_EXPERTS: "144"` in the `run2` arm and add `--max-retries`>0
to re-roll a possible startup OOM at the fragmentation edge.)

## Caveats / future unlocks (would change the recommendation)

- **The b16 wall is the whole story.** If we lift it, a 14.9B model (or chonky experts) at full speed
  becomes the better bet. The cleanest unlock is the **fp8 cast on the EP-dispatch all-gather**
  (`ep_ring.py` — cast hidden states to e4m3 before the all-gather): it **halves the dispatch transient
  buffer** (→ bigger model at b16) *and* the dispatch comm (~1.1× speed). ~10-line fork; untested.
  ragged_all_to_all also avoids the all-tokens-on-all-devices transient but measured a terrible 4.2% MFU.
- **"Less sparsity" (higher K)** raises active params → fewer tokens; K16 variants OOM at b16 anyway.
  spr1's K8 is the validated throughput point; keep it.
- **SFT cooldown loader is flaky** — the spr1 cooldown silently hung twice at ~step 7-8k (a bad SFT
  shard, likely the OpenThoughts-Agent component at weight 0.1). For run #2's cooldown, drop/replace
  that component or pre-build + verify the SFT cache before launch.
