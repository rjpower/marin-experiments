# reentrant

Re-entrant (**self-looping**) transformer experiments on a small (130M-class) Mixture-of-Experts
`grug` model, run as a standalone marin-experiments template. The question: can a weight-tied
model that re-applies its own core layers at inference — "thinking" in activation space — improve
loss by looping more, as an alternative to emitting chain-of-thought *tokens*?

The full write-up — motivation, the E0–E7 experiment series, the depth-scaling sweeps, and the
mechanism behind the negative result — is in [`REPORT.md`](REPORT.md).

**Short answer:** at the d512 compute-optimal point, no looped variant beats the dense baseline
(E0, paloma 3.818), and the failure is structural: a residual core re-applied in place cannot both
keep computing and converge, so extra test-time loops drift off-manifold rather than refine. Four
targeted fixes (contraction penalty, depth-conditioned routing, anytime supervision, learned
halting) each confirm the same diagnosis. See the report for the numbers.

> **Dependencies.** marin is consumed as the nightly `0.2.x.dev` wheels from **PyPI**
> (`marin-core`, `marin-levanter`, …). This experiment deliberately does *not* use the GitHub
> `*-latest` release find-links that the other templates use: those serve `0.99.dev<date>` wheels
> that are stale *and* outrank the fresh PyPI wheels by version, which silently pins old marin code
> (see the comment in `pyproject.toml`). Repin with `uv lock --upgrade`.

## Architecture

The re-entrant model is a **prelude → shared recurrent core → coda** stack. Prelude and coda are
unique blocks; the core is a single block applied `R` times with strict weight-tying. Effective
depth `= 1 (prelude) + R (core loops) + 1 (coda) = R + 2`, so a looped model is compute-matched to
a dense one of the same effective depth while using far fewer unique parameters. The parameter tree
is independent of `R`, so one trained checkpoint can be evaluated at any loop count by swapping the
static `recurrence_steps` (this is what `eval_sweep.py` does).

## Layout

Everything is flat in this directory (no package); `launch.py` is the entry point.

| file | role |
|---|---|
| `launch.py` | builds the `ExecutorStep`, selects the experiment from `GRUG_EXPERIMENT`, runs training in-process on the TPU job |
| `eval_sweep.py` | restores one checkpoint and re-evaluates it at a sweep of recurrence depths R (the test-time depth-scaling experiment) |
| `model.py` | the re-entrant grug MoE model (prelude/core-loop/coda, FiLM, depth-routing, anytime, PonderNet) |
| `train.py` | the grug training loop (`_run_grug_local`), with the randomized-depth / consistency / anytime / ponder loss wiring |
| `data.py` | pins the existing tokenized Nemotron-CC + code caches (training) and Paloma + uncheatable-eval caches (validation) by path — no re-tokenize |
| `heuristic.py` | derives model/optimizer/batch/steps from a compute budget |
| `optimizer.py`, `adamh.py` | base AdamH optimizer config (FiLM params fall back to plain Adam) |
| `dispatch.py`, `checkpointing.py` | grug plumbing |
| `test_eval_sweep.py`, `test_optimizer.py` | CPU unit tests for the depth-sweep plumbing and the optimizer |

## Experiments (arms)

The experiment is selected by `GRUG_EXPERIMENT` (comma-separated names; default `e0`), so one
module submits every variant. All arms share one compute budget / model size (the d512 point) so
curves are directly comparable; only the model architecture changes.

| env value | variant | what it adds vs. its predecessor |
|---|---|---|
| `e0` | dense baseline | 6 independent blocks, R=1 (reference curve) |
| `e1` | re-entrant | 3 unique blocks looped to effective depth 6 |
| `e2` | + FiLM | per-iteration FiLM/adaLN conditioning on the loop index |
| `e3` | + randomized depth | core loop count sampled per step from {2,4,8} (the depth-scaling arm) |
| `e5` | + consistency penalty | training-only core-contraction penalty (`CONSISTENCY_WEIGHT`) |
| `e6` | + depth routing | learned per-(iteration, layer) router-logit bias (f_t ≠ f) |
| `e4` | + anytime supervision | CE read off the shared head after every iteration (`ANYTIME_WEIGHT`) |
| `e64` | e6 + e4 | depth-conditioned routing *and* anytime supervision |
| `e7` | + PonderNet | learned per-token halting head (`PONDER_KL`, `PONDER_PRIOR`) |

## Data

`data.py` does **not** re-tokenize. It pins each component of the Nemotron-CC + starcoder +
proofpile training mixture, and each Paloma + uncheatable-eval **validation** set, to its
already-materialized GCS cache via `with_output_path`, so the executor verifies the cache and skips
tokenization. The headline metric is the **Paloma macro loss**, so unlike the delayed-gradient-pp
template the validation sets *are* attached (weight 0 in the mixture). The tokenizer is
`meta-llama/Meta-Llama-3.1-8B`; the validation cache hashes were resolved with marin's own
`compute_output_path(default_validation_sets(llama3))` and verified against the materialized
caches.

The caches are in `us-central1` (`gs://marin-us-central1`), so **runs must target that region** to
avoid cross-region reads — which also matches the v5p-8 baseline hardware.

## Running

CPU unit tests of the depth-sweep + optimizer plumbing (no cluster):

```bash
JAX_PLATFORMS=cpu uv run pytest
```

Submit one arm directly onto a TPU on the shared marin cluster (training runs in-process on the job
that holds the TPU — no Fray driver hop — when `GRUG_DIRECT` is set):

```bash
GRUG_EXPERIMENT=e3 GRUG_DIRECT=1 MARIN_PREFIX=gs://marin-us-central1 \
  uv run iris --cluster=marin job run --no-wait \
    --tpu v5p-8 --region us-central1 --enable-extra-resources --extra marin-core:tpu \
    --cpu 32 --memory 128GB --disk 50GB \
    -e WANDB_API_KEY "$WANDB_API_KEY" -e GRUG_DIRECT 1 -- python launch.py
```

Test-time depth-scaling sweep over a trained checkpoint (the central experiment): restore once and
re-evaluate at `R = 2,4,8,16,32` by swapping `recurrence_steps`. `SWEEP_MODEL` selects which
variant's model config to restore with (e3/e4/e6/e64/e7); `CHECKPOINT_PATH` is the `gs://`
checkpoint dir:

```bash
SWEEP_MODEL=e3 CHECKPOINT_PATH=gs://marin-us-central1/.../checkpoints/step-6387 \
  RECURRENCE_VALUES=1,2,4,8,16,32 GRUG_DIRECT=1 MARIN_PREFIX=gs://marin-us-central1 \
  uv run iris --cluster=marin job run --no-wait \
    --tpu v5p-8 --region us-central1 --enable-extra-resources --extra marin-core:tpu \
    --cpu 32 --memory 128GB --disk 50GB \
    -e WANDB_API_KEY "$WANDB_API_KEY" -e GRUG_DIRECT 1 -- python eval_sweep.py
```

The sweep prints a `depth-scaling eval sweep` table (R vs. macro/paloma loss) to stdout and logs
per-R metrics to wandb.
