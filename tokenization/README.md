# Tokenizer FLOP-equivalent bake-off

How much real, **compute-fair** uplift does grug-moe get from an alternative tokenizer family?
Quality is bits-per-byte (BPB, tokenizer-agnostic); serving cost is priced at the deployment
target; the two combine into a FLOP-equivalent BPB (feBPB) that credits a denser (cheaper-to-serve)
tokenizer for reinvesting its serving saving into training. The pipeline is deliberately split so
results **re-score under new deployment assumptions without retraining** — training logs raw
`(train-FLOPs, BPB)` and fertility logs raw token/byte counts; the cost model is applied only at
analysis time.

Design, rationale, and the full (corrected) findings — the honest verdict is that SuperBPE is a
serving-cost lever, not a scale-robust quality-per-FLOP win — live in
[marin-community/marin#6796](https://github.com/marin-community/marin/issues/6796).
Per-experiment reproduce commands: [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md).

## The pipeline

Every stage is a marin `ArtifactStep` (`marin.execution.lazy`); `bakeoff.py` wires them into one
DAG and drives the CPU/analysis pieces.

```
corpus            build the raw multi-domain tokenizer-training corpus     corpus.py
  → train_tok     train + push each SuperBPE/BPE tokenizer to mirror://     train_tokenizer.py
  → fertility     measure tokens/byte per arm on the soak corpus           fertility.py
hf_download       mirror each soak data source to S3 (once, shared)        train_model.py
  → tokenize      tokenize each source with each arm's tokenizer           train_model.py
  → train_model   one 24h grug-moe soak per arm, BPB → W&B                 train_model.py
collect_results   pull the BPB ladder + loss curves from W&B              collect_results.py
  + fertility → analysis   score feBPB + raw BPB                           analysis.py
```

| Module | Role |
|---|---|
| `bakeoff.py` | **Entry point.** The arm roster + `prep` / `soak` / `score` subcommands. |
| `arms.py` | Registry of tokenizer arms (`name`, loadable ref, vocab size, design axis). |
| `cost_model.py` | Scoring core: `ServingCostModel`, BPB-curve fit, and `febpb`. Run it directly for a self-check. |
| `corpus.py` | Builds the ~4 GB English/code/multilingual/math raw-text corpus the trained tokenizers learn from. |
| `superbpe.py` | Plain BPE (stock Rust trainer) and SuperBPE (two-stage superword BPE, a from-scratch reimplementation of arXiv:2503.13423). |
| `train_tokenizer.py` | Trains one `TrainSpec` from the corpus and pushes it to `mirror://tokenizers/trained/<name>` so a worker loads it by `trained/<name>`. |
| `fertility.py` | Measures tokens/byte per arm over the soak corpus (or streamed HF eval domains) — the serving-cost pre-filter. |
| `train_model.py` | One 24h grug-moe soak per arm (`soak_checkpoint`): the arm drives both data tokenization and model vocab; held-out multi-domain BPB eval on. |
| `proxy_ladder.py` | The short isoFLOP proxy run (hidden-1024, SlimPajama) — the cheap pre-screen before the soak. |
| `collect_results.py` | Pulls each arm's BPB ladder, per-domain curves, domain-fair macro, and loss/perplexity curves from W&B. |
| `analysis.py` | Re-scores stored fertility + ladder under a `ServingCostModel` chosen on the CLI (context, sparsity, target size, serving ratio, domain mix). |

## Setup

Two dependency tiers:

- **Analysis + prep** (`bakeoff.py`, `analysis.py`, `cost_model.py`, `arms.py`, `corpus.py`,
  `superbpe.py`, `train_tokenizer.py`, `fertility.py`, `collect_results.py`, `soak_config.py`) is
  self-contained on the marin PyPI packages in `pyproject.toml` — `uv sync` and run.
- **The GPU soak launchers** (`train_model.py`, `proxy_ladder.py`) additionally import the grug-moe
  training stack (`experiments.grug.moe`, `experiments.datasets`), which is **not** published to
  PyPI. Run them from a marin source checkout with it on `PYTHONPATH`:

  ```bash
  PYTHONPATH=/path/to/marin:/path/to/marin/lib/levanter/src uv run python train_model.py
  ```

## Run it

```bash
# 0. sanity-check the scoring model (no cluster, no data)
uv run python cost_model.py

# 1. prep — build the corpus, train the fixed-soak tokenizers, measure fertility (CPU steps).
#    Prints the DAG by default; --run builds it.
uv run python bakeoff.py prep --run

# 2. soak — print the per-arm `iris job run` commands for the 24h GPU soaks, then launch and
#    monitor each individually (each is its own long job).
uv run python bakeoff.py soak

# 3. score — pull results from W&B and compute the feBPB analysis.
uv run python bakeoff.py score --run
```

### Re-score under new deployment assumptions

`analysis.py` recomputes the feBPB frontier from the stored measurements — change the cost model
and re-run to answer "what's optimal if we serve at 64k context / a 400B model / a code-heavy
mix?" without touching the cluster:

```bash
uv run python analysis.py \
  --fertility results/rerun_fertility.json --bpb results/rerun_bpb_macro7.json \
  --ref-budget 6e19 --serving-ratio 1          # the corrected soak result at the 6e19 floor
uv run python analysis.py ... --context-len 65536      # serve at 64k
uv run python analysis.py ... --target-hidden 8192 --target-layers 80
```

Steps 1–2 are CPU-heavy; on cw-rno2a submit them as `iris` jobs (leave headroom below the single
node's capacity — the controller shares it). See `EXPERIMENT_LOG.md` for the full soak matrix.
