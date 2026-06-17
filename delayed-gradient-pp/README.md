# delayed-gradient-pp

Pipeline-parallel **staleness** experiments for a small (130M-class) Mixture-of-Experts
`grug` model, run as a standalone marin-experiments template. The base optimizer
is **Muon**, and the question is whether a throughput-optimal (asynchronous)
pipeline schedule — which applies **stale gradients** — is worth adopting for MoE
pretraining, and whether the staleness can be corrected cheaply with `O(weights)`
extra optimizer state.

We study staleness *in software*: instead of building pipeline parallelism, we
inject a controlled per-parameter gradient delay into a single `jax.jit`'d
training step and measure the convergence cost. At delay τ=0 this is bit-identical
to the unwrapped optimizer; for τ>0 it reproduces constant-delay asynchronous SGD
exactly. The delay FIFO and any correction statistics are the `O(weights)` state
we are budgeting for.

> **Dependencies.** marin is consumed as the nightly `0.2.x.dev` wheels from
> **PyPI** (`marin-core`, `marin-levanter`, …). This experiment deliberately does
> *not* use the GitHub `*-latest` release find-links that the other templates
> use: those serve `0.99.dev<date>` wheels that are stale *and* outrank the fresh
> PyPI wheels by version, which silently pins old marin code (see the comment in
> `pyproject.toml`). The model depends on the grug `_moe/` refactor (marin #6312,
> 2026-06-11), so the lock must stay on a `>= 0.2.14.dev202606120949`
> `marin-levanter`. Repin with `uv lock --upgrade`.

## Layout

Everything is flat in this directory (no package); `launch.py` is the entry point.

| file | role |
|---|---|
| `launch.py` | builds the `ExecutorStep`, selects the arm from env vars, runs training in-process on the TPU job |
| `data.py` | pins the existing tokenized Nemotron-CC + code caches by path (no re-tokenize) |
| `model.py` | the grug MoE model config (hidden 512, 64 experts, top-4, 6 layers) |
| `train.py` | the grug training loop (`_run_grug_local`) |
| `heuristic.py` | derives model/optimizer/batch/steps from a compute budget |
| `optimizer.py`, `adamh.py` | base Muon / AdamH optimizer configs |
| `delay_optim.py` | the delayed-gradient wrapper + correctors (the core of the experiment) |
| `dispatch.py`, `checkpointing.py` | grug plumbing |
| `analyze_isoloss.py` | offline iso-loss / token-overhead analysis from wandb |
| `pp_throughput_model.py` | parametric v6e/DCN throughput break-even model |
| `_smoke_delay.py` | CPU smoke test for the delayed wrapper |

## Arms

The arm is selected entirely by environment variables, so one module submits many
times with different staleness / corrector settings:

| env var | meaning | default |
|---|---|---|
| `GRUG_OPT` | `muon` \| `adamh` | `muon` |
| `GRUG_TAU` | uniform gradient delay in steps (ignored if `GRUG_STAGES`>0) | `0` |
| `GRUG_STAGES` | pipeline stages for the per-stage delay profile | `0` (uniform) |
| `GRUG_CORRECTOR` | `none` \| `dc_asgd` \| `dc_asgd_ema` \| `weight_pred` \| `lr_damp` \| `wp_preorth` \| `wp_cautious` \| `wp_trust` \| `wp_confidence` | `none` |
| `GRUG_DC_LAMBDA` | DC-ASGD strength | `1.0` |
| `GRUG_PRED_SCALE` | `weight_pred` / `wp_*` horizon as a multiple of τ | `1.0` |
| `GRUG_PRED_BETA` | `wp_*` raw-momentum EMA decay | `0.95` |
| `GRUG_TRUST` | `wp_trust` trust-ratio clamp | `0.01` |
| `GRUG_LR_DAMP` | `lr_damp` step multiplier | `1.0` |
| `GRUG_STEPS` | train steps | `3000` |
| `GRUG_SEED`, `GRUG_HIDDEN`, `GRUG_BUDGET`, `GRUG_GROUP` | seed / model size / compute budget / wandb group | see `launch.py` |

## Data

`data.py` does **not** re-tokenize. It pins each component of the Nemotron-CC +
starcoder + proofpile mixture to its already-materialized GCS cache via
`with_output_path`, so the executor verifies the cache and skips tokenization. All
caches live under `$MARIN_PREFIX`; the tokenizer is `meta-llama/Meta-Llama-3.1-8B`.
No validation sets are attached — the convergence metric is `train/loss`.

The caches are in `europe-west4`, so **runs must target that region** to avoid
cross-region reads.

## Running

CPU smoke test of the delayed wrapper (no cluster):

```bash
JAX_PLATFORMS=cpu .venv/bin/python _smoke_delay.py
```

Submit a single arm directly onto a TPU on the shared marin cluster (training runs
in-process on the job that holds the TPU — no Fray driver hop):

```bash
GRUG_OPT=muon GRUG_STAGES=6 GRUG_CORRECTOR=weight_pred GRUG_STEPS=6000 \
  MARIN_PREFIX=gs://marin-eu-west4 \
  uv run iris --cluster=marin job run --no-wait \
    --tpu v6e-8 --enable-extra-resources --extra marin-core:tpu \
    --max-retries 3 --cpu 32 --memory 128GB --disk 50GB \
    -e WANDB_API_KEY "$WANDB_API_KEY" -- python launch.py
```

Iso-loss / token-overhead analysis over a wandb group:

```bash
.venv/bin/python analyze_isoloss.py \
    --group delay-pp-isoloss --sync delay-muon-d512-tau0-none-s0-st6000
```

## Key results (130M-class MoE, simulation)

- **Per-stage delay ≠ one global delay.** A faithful 6-stage profile (`pp6`, last
  stage fresh, embedding stalest) plateaus +0.086 above synchronous Muon; treating
  the same model as a uniform τ=5 delay plateaus +0.282 — 3.3× worse. Modeling a
  pipeline as a single delay overstates the cost 2–3×.
- **Staleness is a recoverable token tax, not a quality floor.** Uncorrected Muon
  needs 1.33× the tokens of synchronous training to reach matched loss at 6k
  steps, falling to 1.16× by 15k and still descending. Adam (`adamh`) needs 2.44×.
- **Weight prediction is the corrector that works — and it must extrapolate the
  orthogonalized update.** Evaluating the gradient at `Ŵ = w − τ·lr·NS(M)` cuts the
  tax to 1.23×. Predicting along the *raw* momentum `M` (pre-Newton-Schulz) is
  worse than no correction: NS substantially rotates the direction, so the forward
  evaluation lands at a point the optimizer never visits. DC-ASGD curvature
  correction recovers ≤3% and is inert on Muon.
- **Throughput verdict.** A v6e/DCN model turns the converged 1.16× tax into a net
  1.08–1.73× useful speedup once cross-slice gradient traffic is exposed (DCN
  comm/compute ≥ 0.5), and a wash when deeply compute-bound.

These are simulation results; no real pipeline schedule has been run yet. Full
tables, per-run links, and the corrector sweep are in the tracking issue
([marin #6431](https://github.com/marin-community/marin/issues/6431)).
