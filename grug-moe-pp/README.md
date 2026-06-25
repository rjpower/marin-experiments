# grug-moe-pp

Pipeline parallelism (toward **zero-bubble** / 1F1B) for the **production**
`grug` Mixture-of-Experts model, composing with expert parallelism (EP) and
FSDP on both TPU and GPU. PP is the final parallelism axis layered on top of the
existing EP/FSDP grug stack.

This is the working snapshot of the investigation, moved out of marin for the
record. The design notes are in [`NOTES.md`](NOTES.md); the headline result is
below.

## Status: parked

Short version — **it works but it doesn't pay off at this scale, so we're
closing it out for now.**

- A gradient-exact, **bubble-free** PP for the production grug-MoE does run on
  real hardware (v6e-8 / v6e-32) and is grad-exact vs the non-pipelined oracle.
- It delivers the **memory-scaling** win: PP trains batches that OOM FSDP — an
  8M-token batch fits in ~22.8 GiB under 1F1B, where the FSDP baseline OOMs at
  1M tokens on the logit head (~137 GiB).
- But it does **not** beat FSDP on **throughput** at the 100M–1B / v6e-8 scale
  studied: best bubble-free PP ≈ 22.9k tok/s vs FSDP ≈ 145k tok/s on v6e-8
  (~0.78× single-host). PP only pulls ahead once the FSDP parameter all-gather
  has to cross DCN — on v6e-32 it's ~1.22× FSDP.
- The residual gap is MFU + recompute, **not** pipeline bubble, memory, or
  Python-overhead — so there is no easy schedule fix left to chase.

Full measurements and the milestone log live in the original marin work:
PR [marin-community/marin#6534][pr] and issue [marin-community/marin#6532][issue].

[pr]: https://github.com/marin-community/marin/pull/6534
[issue]: https://github.com/marin-community/marin/issues/6532

## Dependencies — archived snapshot, not a wheel-only template

Unlike the other entries in this repo, this directory is **not** self-contained
against the published `marin-*` wheels. Running any of it requires a **marin
source checkout**, because the code imports:

- `experiments.grug.moe.model` — the production grug-MoE `Transformer` /
  `GrugModelConfig`. This lives in marin and is **not** vendored here.
- `levanter.grug.*` — the grug EP / FSDP / sharding internals, **including a
  change that was never merged**: `compact_grug_mesh` and `_GRUG_MESH_AXIS_NAMES`
  gain an outermost `stage` mesh axis so PP can manualize a `shard_map` over
  `stage` while the other axes stay GSPMD-partitioned. That change is preserved
  as [`levanter-stage-axis.patch`](levanter-stage-axis.patch).

To run a check or benchmark, from a marin checkout with the patch applied:

```bash
# in lib/levanter, apply the unmerged mesh change
git apply /path/to/grug-moe-pp/levanter-stage-axis.patch   # touches lib/levanter/.../grug/sharding.py

# then, from inside this directory, with marin on PYTHONPATH so
# `experiments.grug.moe` and `levanter` resolve:
PYTHONPATH=/path/to/marin:/path/to/marin/lib/levanter/src uv run python check_zb.py
```

The intra-directory imports are flat (`from pipeline_zb import ...`); the two
imports above are the only external marin dependencies.

## Layout

| file | role |
|---|---|
| `pipeline.py` | **the kept TPU baseline** — production `Transformer` in one `shard_map` that manualizes every mesh axis (`{stage, expert, data}`), differentiated with whole-program `value_and_grad`. GPipe schedule ripples activations stage→stage+1 via `ppermute`; real sparse ring-EP inline (all_gather dispatch + megablox `ragged_dot` GMM + `psum_scatter` collect); FSDP shards weights over `data` with a custom-vjp `all_gather` that keeps the weight grad `/data`-sharded. |
| `pipeline_zb.py` | zero-bubble wavefront (ZB-H1) over the same stages — `Schedule` enum `{gpipe, 1f1b, zb}`, `TransportMode` enum, Muon (Newton-Schulz) gradient amortization, multi-host `ppermute` transport. |
| `pipeline_manual.py` | manual per-stage `jax.vjp` backward — the GPU OOM-dodge: avoids the `[num_stages, …]` weight-grad buffer that whole-program autodiff stacks. |
| `oracle.py` | non-pipelined reference loss/grad on the `stage=1` mesh, for the parity checks. |
| `check_zb.py` | gradient-exactness: `zb_value_and_grad` vs the oracle across stage / expert / data splits. |
| `check_manual_pp.py` | gradient-exactness: `manual_pp_value_and_grad` vs the oracle. |
| `benchmark.py` | PP vs FSDP / EP forward+backward timing harness (drives `pipeline.py`); shared `_config` / param-count / peak-HBM helpers. |
| `perf.py` | manual-PP timing harness. |
| `perf_zb.py` | zero-bubble timing harness (sweeps `Schedule` / `TransportMode` / remat). |
| `gpu_smoke.py` | GPU make-or-break smoke — does the per-stage manual backward dodge the OOM at scale? |
| `overlap_probe.py`, `kernel_overlap_probe.py` | diagnose pipeline / dispatch overlap (serialization). |
| `ppermute_probe.py` | cross-host GPU↔GPU `ppermute` transport make-or-break. |
| `thread_probe.py` | threaded async host-transport probe. |
| `NOTES.md` | design notes: what's kept, what was tried and dropped, and the intended direction. |
