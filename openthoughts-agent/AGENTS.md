# openthoughts-agent — agent notes

Goal: build the best **OpenThoughts terminal agent** on **Qwen3-8B**
(https://www.openthoughts.ai/blog/agent), trained with `google-tunix` and run on
the **marin/iris** TPU cluster (v6e-8 / v6e-16 slices). Read `/AGENTS.md` (repo
shared practices) first; only experiment-specific conventions are here.

## Pipeline (where we are)

1. **SFT** Qwen3-8B on `open-thoughts/OpenThoughts-Agent-v1-SFT` (~15.2k
   Terminus-2 agent trajectories) → checkpoint to GCS. ← *current milestone*
2. **Eval** the SFT'd agent on `open-thoughts/OpenThoughts-TB-dev` /
   Terminal-Bench. Tasks are Docker environments + graders, so tool execution
   runs in a **gvisor (runsc) sandbox** inside the (privileged) TPU task.
3. **RL** (Dr.GRPO via tunix) on the ~720 verified RL tasks, sync rollouts first,
   then async.

## Source layout

- `models/qwen3_loader.py` — load a stock HF Qwen3 into tunix's native `flax.nnx`
  Qwen3. Data-driven from `config.json`; threads `param_dtype` (fp32 for SFT),
  `remat` (DECODER for 8B), `use_flash_attention`. NO RoPE patch (Qwen3-8B is a
  standard Qwen3: `rope_theta=1e6`, no scaling).
- `models/registry.py` — name → repo + loaders. `qwen3-8b` (chat, the agent base),
  `qwen3-8b-base`, `qwen3-1.7b-base` (fast smoke).
- `data/agent_traces.py` — stream `OpenThoughts-Agent-v1-SFT` (pinned revision) as
  `{messages, metadata}`. Schema: `conversations: List[{role, content}]`.
- `training/agent_sft.py` — ChatML encoder + **assistant-turn loss masking** +
  the `PeftTrainer` SFT loop with **orbax checkpointing**.
- `training/common.py` — `clipped_adamw`, `build_mesh(tp=...)`, `sft_model_input_fn`.
- `launch_sft.py` — iris entrypoint (stage 1).

## Running on iris

`iris` is invoked via `uv run iris` from this dir (it depends on `marin-iris`).
Cluster config is the packaged `--cluster=marin` lookup (NOT a local file). Auth
tokens are cached in `~/.iris/`; `iris login` if missing.

**Submit the SFT (v6e-16, fp32 8B actor + AdamW):**

```bash
uv run iris --cluster=marin job run --no-wait \
  --tpu v6e-16 --enable-extra-resources --extra tpu --region europe-west4 \
  --cpu 8 --memory 200GB --disk 200GB --max-retries 1 --job-name ota-sft-qwen3-8b \
  -e HF_TOKEN "$HF_TOKEN" -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e CKPT_DIR gs://<bucket>/openthoughts-agent/qwen3-8b-sft \
  -- python launch_sft.py
```

**Fast plumbing smoke (small model, tiny data, v6e-8):**

```bash
uv run iris --cluster=marin job run --no-wait \
  --tpu v6e-8 --enable-extra-resources --extra tpu --region europe-west4 \
  --cpu 8 --memory 120GB --disk 80GB --max-retries 1 --job-name ota-sft-smoke \
  -e HF_TOKEN "$HF_TOKEN" -e AGENT_MODEL qwen3-1.7b-base \
  -e SFT_STEPS 20 -e DATA_LIMIT 256 -e MAX_SEQ_LEN 2048 \
  -- python launch_sft.py
```

**gvisor eval/RL** (later milestones) needs a custom task image with `runsc`,
shipped to GHCR and selected per-job with `--task-image`:

```bash
uv run iris --cluster=marin job run --task-image ghcr.io/<org>/ota-agent-task:latest ...
```

### Job knobs (env)

`AGENT_MODEL`, `SFT_STEPS`, `BATCH_SIZE`, `LR`, `MAX_SEQ_LEN`, `SEED`, `TP`,
`DATA_LIMIT`, `CKPT_DIR`, `EVAL_TOKENS` — see `launch_sft.py` docstring.

## Hard-won invariants (carry these or re-break them)

- **Commit before submit.** iris bundles the dir via `git-ls-files`; uncommitted
  changes do NOT ship. `.venv`/`__pycache__` excluded; 25 MB bundle cap.
- **`--region` is mandatory.** Default region has no TPU capacity. v6e zones:
  `europe-west4`, `us-east1`, `us-east5`.
- **`launch_*.py` must stay at the repo (experiment) root** and run as
  `python launch_sft.py` (NOT `-m`), so the iris submit command resolves.
- **fp32 actor params for SFT.** A 1e-5–8e-5 AdamW update is below bf16 ULP for
  unit-scale weights → bf16 storage silently zeroes updates. Compute in bf16.
- **Gradient clipping is load-bearing**, not cosmetic: unclipped → inf/NaN grad →
  libtpu SIGSEGV that kills the run. Always `clipped_adamw`.
- **tunix `Sampler` decodes GREEDILY unless `top_p` is passed** (it ignores
  `temperature` AND `seed`). Pass `top_p=1.0` to every eval/rollout sampler.
- **Use grain, not HF `.batch()`** — tunix's `jax.tree.map(np.repeat, ...)`
  corrupts HF-batched rows.
- **`kv_cache_size >= max_prompt_length + max_tokens_to_generate`** or the sampler
  hard-errors; keep headroom.
- **One model family per process** (only matters if mixing Delphi + Qwen3; we
  only load Qwen3 here, so no RoPE-patch hazard).
- **Re-`uv lock` before submit if stale** — `marin-*` are nightly `0.2.x.dev`
  wheels that drift.
- **Host RAM was often the limiter.** Use `--memory 200GB` for 8B-class jobs.

## Dependencies

`google-tunix` (0.1.7 from PyPI; has the `qwen3_8b` preset, `PeftTrainer` +
orbax checkpointing, Dr.GRPO), `marin-iris`, `marin-fray`. TPU extra
(`--extra tpu`) pulls `jax[tpu]`/libtpu on the worker. Local async-RL work may
later switch to the `rjpower/tunix` fork (`~/code/tunix`).

## Local checks (CPU)

```bash
uv run pytest -q                       # fast encoder/masking tests (tokenizer only)
uv run pytest -q -m slow               # model/JAX tests (needs weights/accel)
```
