# adaptive-sparsity

**Strong sparsity** experiments for a ~1B-class Mixture-of-Experts `grug` model, run
as a standalone marin-experiments template. We ask two questions:

1. **Baseline sparsity (1–5% active).** How does prediction quality move as we vary
   the fixed active-expert fraction `K/E` at a matched token budget? The MoE
   literature predicts loss is roughly flat-to-improving as you sparsify at fixed
   compute (Abnar et al., [2501.12370](https://arxiv.org/abs/2501.12370)); we test
   it directly on a 1.1B-total / ~150M-active grug MoE.

2. **Adaptive / aggressive sparsity (0.1% → 5%).** Instead of a fixed `K`, let the
   router pick a *variable* number of experts per token, and add a **soft sparsity
   penalty** that conditions the model toward fewer active experts wherever the
   cross-entropy loss still permits. How far down can the average active fraction go
   before quality breaks?

This is the sister experiment to [`delayed-gradient-pp`](../delayed-gradient-pp) and
reuses its grug training plumbing. The full writeup lives in
[`REPORT.md`](REPORT.md) once the runs land.

## The adaptive routing mechanism

The base grug MoE routes each token to a fixed top-`K` of `E` experts, with
DeepSeek-style loss-free (bias) balancing and a float32 router z-loss — the three
stability mechanisms the literature deems mandatory for high sparsity are already
present. We add **variable-k threshold routing** (`model.py`, `MoEMLP._adaptive_gate`):

- Select the top-`K_max` candidates as usual (`K_max = num_experts_per_token`).
- Keep only those whose biased router logit clears a **learned per-layer threshold**
  `θ`; always keep the strongest `min_experts_per_token` (the floor; `0` lets an
  easy token fall through to the always-on shared expert alone). Dropped slots get a
  zero combine weight.
- Penalize the **expected (soft) active fraction** `coef · E[Σ sigmoid((logit−θ)/temp)] / E`.
  Minimizing it competes with cross-entropy: the model drops an expert only where
  the prediction does not need it.
- A **straight-through estimator** keeps the forward pass *truly* sparse (hard keep)
  while routing the threshold's gradient through the soft surrogate — without it `θ`
  sees only the penalty and collapses to the floor.
- Training **starts dense** (θ inits below the logit scale, so all `K_max` slots
  clear it) and the penalty anneals it sparse, à la ReMoE
  ([2412.14711](https://arxiv.org/abs/2412.14711)).

> **Cost accounting.** The grug dispatch kernel has a fixed `[T, K_max]` shape, so a
> dropped slot is masked (zero combine weight) rather than skipped — the forward is
> *quality*-sparse but still pays `K_max` expert FLOPs. We therefore report the
> realized active fraction and the **quality↔sparsity frontier**; turning that into
> wall-clock FLOP savings needs a ragged/variable-capacity kernel (a separate systems
> change), exactly as `delayed-gradient-pp` framed its delay study as a simulation.

## Layout

Flat directory (no package); `launch.py` is the entry point.

| file | role |
|---|---|
| `launch.py` | builds the `ExecutorStep`, selects the arm from env vars, runs training in-process on the TPU job |
| `model.py` | the grug MoE model + adaptive variable-k routing and the soft sparsity penalty |
| `data.py` | region-agnostic FineWeb-Edu data (HF-backed pre-tokenized cache) |
| `heuristic.py` | derives model/optimizer/batch/steps from a compute budget (E/K and the adaptive knobs are now overridable) |
| `train.py` | the grug training loop (`_run_grug_local`) |
| `optimizer.py`, `adamh.py` | AdamH optimizer configs |
| `dispatch.py`, `checkpointing.py` | grug plumbing |
| `_smoke_sparsity.py` | CPU smoke test for the routing + adaptive gate + penalty |

## Arms

The arm is selected entirely by environment variables, so one module submits every
arm of the sweep. Every arm shares the same batch size and step count (**iso-token**,
~2.7B tokens at the default budget) and the same AdamH learning rate, so the only
variable across arms is the routing sparsity.

| env var | meaning | default |
|---|---|---|
| `SPARSITY_MODE` | `fixed` \| `adaptive` | `fixed` |
| `SP_HIDDEN` | model hidden dim (768 → ~1.1B total at E=128) | `768` |
| `SP_EXPERTS` | number of experts `E` | `128` |
| `SP_TOPK` | top-k width `K` (= `K_max` capacity in adaptive mode) | `4` |
| `SP_MIN_K` | adaptive per-token floor | `0` |
| `SP_COEF` | sparsity penalty weight `λ` (adaptive only) | `0.0` |
| `SP_TEMP` | soft keep-gate temperature | `1.0` |
| `SP_BUDGET` | compute budget for sizing / LR | `1.7e18` |
| `SP_STEPS`, `SP_BATCH` | override steps / batch | heuristic |
| `SP_SEED`, `SP_TPU`, `SP_GROUP`, `SP_SMOKE` | seed / TPU type / wandb group / 10M-token smoke subset | see `launch.py` |

## Data

`data.py` uses the **HuggingFace-backed** pre-tokenized FineWeb-Edu cache
(`marin-community/fineweb-edu-pretokenized-10B`, ~10B tokens). Its source of truth is
the HF Hub, not one GCS region, so the executor re-downloads it into whichever
region's `$MARIN_PREFIX` bucket the TPU job lands in — **the run is not region-locked**
(unlike `delayed-gradient-pp`'s europe-west4-pinned Nemotron caches). Take v6e
capacity wherever it is by omitting `--region`. The tokenizer is the llama3-equivalent
`marin-community/marin-tokenizer` (vocab 128256). No validation sets are attached; the
convergence metric is `train/loss` at matched tokens.

## Running

CPU smoke test of the routing + adaptive gate (no cluster):

```bash
JAX_PLATFORMS=cpu uv run python _smoke_sparsity.py
```

Submit one arm onto a TPU on the shared marin cluster (omit `--region` to take v6e
capacity anywhere; the data cache is region-agnostic):

```bash
MARIN_PREFIX=gs://marin-us-east5 \
SPARSITY_MODE=adaptive SP_EXPERTS=128 SP_TOPK=8 SP_MIN_K=0 SP_COEF=3 \
  uv run iris --cluster=marin job run --no-wait \
    --tpu v6e-16 --enable-extra-resources --extra marin-core:tpu \
    --max-retries 3 --cpu 32 --memory 128GB --disk 50GB \
    -e WANDB_API_KEY "$WANDB_API_KEY" -- python launch.py
```

## Key results

See [`REPORT.md`](REPORT.md) and the tracking issue. (Populated as the milestones land.)
