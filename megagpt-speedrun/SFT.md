# SFT cooldown data path — root cause of the silent hang + the robust fix

## TL;DR
The SFT cooldown (`iris_jobs.py cool`) used to **silently hang** at ~step 9.8k: the log filled
with `Data loading is taking a long time: N seconds. Waiting for 1024 items.` (climbing to
**16,310 s ≈ 4.5 h**), the step froze, the job stayed `state=running`, and `--max-retries` could
not recover it. It is now fixed and validated. Launch the run2 cooldown with:

```bash
uv run python iris_jobs.py cool --tag run2cool --init-from <run2 final /checkpoints dir>
```

## Root cause (it was NOT the inline tokenization)
Post-mortem of `cool2-sft` (logs):
- All 3 component caches **built and consolidated fine** — `tulu3`/`smoltalk`/`ot_agent`
  finished at 05:23 / 05:34 / 05:42 — **~20 min BEFORE** the stall began (~06:01). So the inline
  zephyr cache-build was innocent.
- The stall was on the **READ side**, in the data loader's `data_store.get_batch(indices)`
  (`levanter/data/loader.py:451`), which reads token slices from the TreeCache via tensorstore
  (`GreedyPrepackedDataset.get_batch` → `store.data[start:end].read()`).
- The trigger: `cool2-sft` was launched via `iris_jobs.py`, whose `BASE` env **lacked
  `LEVANTER_TS_CACHE_LIMIT`** (default **1 GB**). `launch_cw.sh` sets it to **32 GB** *specifically
  to prevent this* — see its own comment: "chunks get evicted before their 64 seqs are consumed →
  R2 re-fetch thrash → the periodic data-loader stalls (and, worst case, **a hung R2 GET**)".
- The default block-shuffle window (`io_block_size=256 × window_blocks=512`) has a **~2 GB working
  set per component**; with a 1 GB read cache it is evicted and re-fetched from R2 every window. The
  **tensorstore s3 driver has no read timeout** (`build_kvstore_spec`), so when one R2 GET hangs,
  `get_batch()` blocks **forever** → silent, unrecoverable stall.

This is the same failure class that bit the pretrain once (Session 2). The reliable pretrain runs
because `launch_cw.sh` carries the read cache; the `cool` path did not.

## The fix (defense in depth) — `data.py` + `iris_jobs.py`
1. **Static cache, no inline build** (`data.build_sft_mix`, `auto_build_caches=False`,
   `source=None`): read the **pre-built** TreeCache directly — no zephyr cache-build sub-job inside
   the training process at all. (Rebuild the cache with `data.build_sft_cache_build_mix()` only if
   the mixture changes; verify each `<name>/train/shard_ledger.json` has `is_finished: true`.)
2. **Localize to local disk** (`data._localize_sft_cache`, default `SP_SFT_LOCAL=1`): the whole
   consolidated cache is only **3.3 GB** (tulu3 1.44 + smoltalk 1.78 + ot_agent 0.10), so at startup
   we mirror `shard_ledger.json` + `input_ids/` + `assistant_masks/` from R2 to the worker's local
   disk (fsspec, ~30 s–2 min cold; idempotent via a `.localized` marker for `--max-retries`
   resumes) and read it via the tensorstore **`file` driver** → **zero R2 reads during training →
   a hung R2 GET is structurally impossible.** (fsspec, unlike tensorstore's raw s3 driver, has
   request timeouts/retries, so a download problem **crashes loudly + is retryable**, never silent.)
   `SP_SFT_LOCAL=0` disables this and reads R2 directly, relying on (3).
3. **`LEVANTER_TS_CACHE_LIMIT=32 GB` in `iris_jobs.py BASE`** (belt + braces; matches
   `launch_cw.sh`). Holds any real-data working set in RAM after first touch. No-op for
   `SP_SYNTH_DATA=1` benchmarks; not part of the `run_id`, so it never affects cache identity.

## The pre-built cache (on R2, complete)
`s3://marin-na/marin/tokenized/megagpt_sft_v2/<name>/train` — all 3 components consolidated
(`is_finished=True`): `tulu3` 939,343 rows, `smoltalk` 1,043,917 rows, `ot_agent` 15,209 rows.
Fields `input_ids` + `assistant_masks` (assistant-only loss via the chat template's
`{% generation %}` region). Mixture weights 0.5 / 0.4 / 0.1.

## Validation evidence (offline, CPU)
- **Data-loader soak:** iterated the *exact* cooldown data path (static mix, `auto_build_caches=
  False`, packed+block-shuffled mixture, real `train.py` `DataLoader`) for **12,000 steps**
  (batch 16, seq 4096) reading from local disk — **NO stall**, sailing past cool2's ~9858 hang
  point. Every sampled batch had a non-trivial assistant-loss mask (~0.54 of tokens) and in-range
  tokens `[0, 128009] < 128256`. (`scratchpad/validate_sft_loader.py`.)
- **`_localize_sft_cache` end-to-end:** real R2 download of `ot_agent` into a fresh dir (2.3 s),
  marker written, second call skips in 0.00 s; the localized cache is loadable
  (`is_finished=True`, 15,209 rows) and readable via the file driver. `build_sft_mix()` resolves
  (via the executor) to the local paths with `source=None`, `auto_build_caches=False`.

## Geometry (`cool` sweep matches run2 for the param graft)
`E128 / K8 / D1536 / d_e512 / I768 / 16L / seq4096 / b16 / EP8 / save_moe` — identical to the
`run2` arm so `SP_INIT_FROM` grafts the model params cleanly. `SP_TOKENS=13.5e9` reproduces run2's
pretrain peak LR (heuristic derives LR from token magnitude); `SP_SCHEDULE=linear SP_MIN_LR=0`
decays peak→0; `SP_STEPS=20000` ≈ 1.31 B SFT tokens (~2 epochs of the cache / ~10 % of the pretrain
budget / ~2 h at run2's ~178 K tok/s). Bump `SP_STEPS` for a longer cooldown.
