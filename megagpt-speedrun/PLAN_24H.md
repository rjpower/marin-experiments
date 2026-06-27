# 24h production plan (reflection)

Goal: best perplexity/bpb in a fixed 24h wall-clock on 8×H100, then a chat-capable model via
an SFT cooldown. This is the "reflect on how to extend to 24h" step.

## Budget split
- **~20h pretrain** + **~3.5h SFT cooldown** + ~0.5h slack (compile, checkpoint, eval).
- E64/b16 sustains ~1.6 it/s clean = ~105k tok/s ⇒ **~7.5B tokens** in 20h.
  (E256/b8 ≈ ~82k tok/s ⇒ ~5.9B tokens; see config decision.)

## Config decision (E64/b16 vs E256/b8)  — *pending the cwobject sweep, leaning E64/b16*
- We are **token-starved** (NanoGPT-speedrun regime). At E256 active ≈0.75B params, 7B tok ≈
  9 tok/param; at E64 active is far smaller ⇒ many more tok/param ⇒ less undertrained.
- E256 has ~6% better **loss-per-token**, BUT: it can't use b16 (OOMs), so it's stuck at b8 ⇒
  ~0.78× the tokens/s of E64/b16. On the steep part of the loss curve, **more tokens (E64)
  most likely beats better-loss/token (E256)**.
- **Lean: E64/b16** (proven clean on R2 as spr1; fits; fastest). The cwobject sweep
  (cwA E64/b16, cwB E256/b8, cwC E128/b16, cwD E64/b32, same data) confirms or overrides.

## Schedule (WSD)
- Pretrain: `SP_SCHEDULE` cosine/linear decay, `SP_WARMUP≈0.02`, decay to a **low but non-zero**
  `SP_MIN_LR` (e.g. 0.1) over the run. `SP_TOKENS≈7.5e9` sets LR magnitude + horizon;
  `SP_FIT_BATCH=16`. Measure perplexity/bpb at pretrain end → **headline number**.
- SFT cooldown (separate phase, ~3.5h): `SP_DATA=sft` (tulu3+smoltalk+OpenThoughts-Agent,
  assistant-only loss), `SP_INIT_FROM=<pretrain .../checkpoints>`, `SP_SCHEDULE=linear`,
  `SP_MIN_LR=0`, small `SP_REWARMUP`. Yields the chat model without destroying the perplexity
  result (which is taken at pretrain end).
- Rationale for two phases (not a combined SFT-during-cooldown): keeps the perplexity headline
  on pretrain data clean while still delivering a chat model.

## Data
- **Production pretrain: R2 full nemotron mix** (`SP_DATA=datakit`, `build_nemotron_datakit_mix`,
  7 quality-weighted components) — proven clean **solo** with the cache fix
  (`LEVANTER_TS_CACHE_LIMIT=32GB`). A single run does not hit the 4×-concurrent R2 contention.
- **Experiments: cwobject** (`SP_DATA=cw`, cluster-local, fast, 4 concurrent) — that's what the
  mirror + cw_patch enable.
- (Optional: if a nemotron tier finishes mirroring, production could move to cwobject too.)

## Guardrails
- `LEVANTER_TS_CACHE_LIMIT=34359738368`, deep prefetch (baked into launchers).
- Hang watcher (`ops.md`): kill+relaunch on a silent data-loader stall (`--max-retries` won't
  recover it). `--max-retries 3` for genuine crashes (resumes from the 30-min checkpoint).
- Checkpoint frequently (rolling) so cooldown can `SP_INIT_FROM` the latest and an iris retry
  resumes.

## Launch sketch — ACTUAL commands in flight (2026-06-26)
```
# PRETRAIN — LAUNCHED 17:07Z as megagpt-prod-pre (E64/b16, R2 nemotron, cosine-to-0):
./launch_cw.sh megagpt-prod-pre 7000000000 "" megagpt-prod \
  -e SP_EXPERTS 64 -e SP_FIT_BATCH 16 -e SP_SCHEDULE cosine -e SP_WARMUP 0.02 -e SP_MIN_LR 0.0
# -> run-id sparsity-fixed-d1536-E64-k8-de512-s0-st106812-f5d380 ; 106812 steps (~18.5h).
# CHECKPOINTS (R2):
#   s3://marin-na/marin/grug/sparsity/sparsity-fixed-d1536-E64-k8-de512-s0-st106812-f5d380/checkpoints

# SFT COOLDOWN — run AFTER prod-pre completes (init weights-only from its checkpoints dir):
./launch_cw.sh megagpt-prod-sft 700000000 "" megagpt-prod \
  -e SP_EXPERTS 64 -e SP_FIT_BATCH 16 -e SP_DATA sft \
  -e SP_SCHEDULE linear -e SP_MIN_LR 0 -e SP_REWARMUP 0.03 \
  -e SP_INIT_FROM s3://marin-na/marin/grug/sparsity/sparsity-fixed-d1536-E64-k8-de512-s0-st106812-f5d380/checkpoints
# NB: SP_INIT_FROM grafts weights only (fresh optimizer, step 0); SAME geometry required.
# Before launching SFT: smoke-test SP_DATA=sft tokenizes (tulu3+smoltalk+OpenThoughts-Agent -> R2).

# POST-HOC perplexity/bpb headline (load final ckpt, eval on held-out paloma c4_en/wikitext_103):
#   python -m levanter.main.eval_lm --config.checkpoint_path=<ckpt> \
#     --config.data.tokenizer=marin-community/marin-tokenizer \
#     --config.data.components.paloma_c4_en.cache_dir=<paloma c4_en cache>  (see eval notes)
```

## Open items before launch
1. Confirm cwobject training path (cwA2) → run the 4-way sweep → finalize config.
2. Pick final `SP_TOKENS`/`SP_MIN_LR` from the chosen config's measured it/s.
3. Decide held-out eval set for the perplexity/bpb headline.
