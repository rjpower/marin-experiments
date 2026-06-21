# openthoughts-agent — agent notes

Goal: build the best **OpenThoughts terminal agent** on **Qwen3-8B**
(https://www.openthoughts.ai/blog/agent), trained with `google-tunix` and run on
the **marin/iris** TPU cluster (v6e-8 / v6e-16 slices). Read `/AGENTS.md` (repo
shared practices) first; only experiment-specific conventions are here.

## Status (2026-06-21)

| Stage | What | State |
|------|------|-------|
| 1. SFT | Qwen3-8B on `OpenThoughts-Agent-v1-SFT` (~15.2k Terminus-2 traces) → GCS checkpoint | **training on v6e-16** (`/power/ota-sft-qwen3-8b`); plumbing validated end-to-end on a 1.7B smoke |
| 2. Eval | run the SFT'd agent on `OpenThoughts-TB-dev` / Terminal-Bench in a gVisor sandbox | harness built + CPU-tested; **gVisor isolation CONFIRMED live on a v6e TPU task**; full integration run pending the 8B checkpoint |
| 3. RL | Dr.GRPO (tunix), sync rollouts then async | not started |

The whole SFT→checkpoint→restore path is proven; the gVisor sandbox the eval
depends on is proven (see "gVisor sandbox" below). The remaining stage-2 work is
wiring the real checkpoint + TB task images through `launch_eval.py` at scale.

## Pipeline architecture

**SFT.** `launch_sft.py` → loads HF Qwen3 into tunix's native `flax.nnx` Qwen3
(fp32 params, bf16 compute, decoder remat) → ChatML-encodes traces with
**assistant-turn loss masking** → `PeftTrainer` with **orbax** checkpointing to a
`gs://` dir.

**Eval (agentic).** `launch_eval.py`, per TB task:
`build_image(task env)` → `GvisorContainerSandbox(image)` (`docker run
--runtime=runsc`) → `run_episode(model_fn, sandbox, instruction)` drives the
**Terminus-2** loop (model emits `{analysis, plan, commands:[{keystrokes,...}]}`
JSON; we exec each command in the sandbox and feed back terminal output) →
`grade_task` runs the task's `tests/test.sh` and reads the reward file. The
policy is the SFT'd Qwen3 wrapped as a `model_fn` by `eval/model_serving.py`
(tunix Sampler, `top_p=1.0`).

## Source layout

- `models/qwen3_loader.py` — load stock HF Qwen3 → tunix `flax.nnx` Qwen3.
  Data-driven from `config.json`; threads `param_dtype` (fp32 SFT), `remat`
  (DECODER for 8B), `use_flash_attention`. Raises on rope_scaling (Qwen3-8B is
  standard: `rope_theta=1e6`, untied, 36L/4096/32h/8kv/head128). Hard
  key-coverage + all-params-concrete asserts.
- `models/registry.py` — name → repo + loaders. `qwen3-8b` (chat; the agent base),
  `qwen3-8b-base`, `qwen3-1.7b-base` (fast smoke).
- `models/checkpoint.py` — `restore_sft_model(base_dir, ckpt_dir, mesh, ...)`:
  load base (remat=NONE so the sampler's KV-cache works) then orbax
  `maybe_restore`. Validated on CPU against the smoke checkpoint.
- `agent_data/agent_traces.py` — load `OpenThoughts-Agent-v1-SFT` (pinned
  revision) as `{messages, metadata}`. Reads parquet shards via
  `snapshot_download` + **pyarrow directly** (datasets<4 can't parse the saved
  `List` feature type). NOTE: the dir is `agent_data/`, NOT `data/` — the repo
  root `.gitignore` ignores `data/`, which would silently drop it from the iris
  bundle.
- `training/agent_sft.py` — ChatML encoder (`render_chatml`,
  `encode_agent_conversation` with assistant-content+`<|im_end|>`+`\n` loss mask),
  grain source, `run_agent_sft(... checkpoint_dir ...)` PeftTrainer loop.
- `training/common.py` — `init_distributed()`, `clipped_adamw`,
  `build_mesh(tp=...)`, `sft_model_input_fn`.
- `eval/sandbox.py` — **the sandbox layer** (see below). `ensure_sandbox_runtime`,
  `ensure_dockerd`, `build_image`, `GvisorContainerSandbox`, `LocalUnsafeSandbox`,
  `make_sandbox`.
- `eval/agent_loop.py` — `SYSTEM_PROMPT` (Terminus-2), `parse_action` (last
  balanced JSON object with a `commands` key), `run_episode`.
- `eval/tb_tasks.py` — load `OpenThoughts-TB-dev` tasks (env Dockerfile +
  instruction + grader).
- `eval/grade.py` — run `tests/test.sh`, parse reward → pass/fail.
- `eval/model_serving.py` — SFT'd model → `model_fn(messages)->str`.
- `eval/gvisor_smoke.py` — **live gVisor validation** in a privileged TPU task.
- `launch_sft.py` / `launch_eval.py` — iris entrypoints (stages 1 / 2).
- `docker/Dockerfile.agent-task` — custom iris task image (runsc + docker).

## gVisor sandbox (the agent runs untrusted shell commands — isolate them)

Each TB task is its own Docker environment; we run it under the **runsc** OCI
runtime so the agent's commands hit gVisor's emulated kernel, not the host.
**CONFIRMED working** inside an iris TPU task: a `--runtime=runsc` container
reports kernel `4.19.0-gvisor` + dmesg `Starting gVisor...` while the host is
`6.8.0-gcp`.

This works because **TPU tasks run `--privileged`** (iris adds it for
accelerators) → uid=0 + caps for rootful gVisor + a task-local dockerd. A
**CPU-only** iris task is NOT privileged, so the sandbox only works on a `--tpu`
slice (the eval needs the TPU for the policy anyway; for a pure sandbox test use
the smallest slice, v6e-4).

**Two ways to get the runtime in**, and `ensure_sandbox_runtime()` handles both:
1. the custom image `ghcr.io/rjpower/openthoughts-agent-task:latest` (built from
   `docker/Dockerfile.agent-task`) **bakes in** docker + runsc + daemon.json →
   bootstrap is a no-op. Select per-job with `--task-image`. (iris rewrites
   `ghcr.io/...` → the per-continent AR mirror, a pull-through cache, so the GHCR
   package must be **public** — it is.)
2. on the **stock iris image**, `ensure_sandbox_runtime()` downloads Docker's
   static binaries + runsc at runtime (same as the smoke). So the eval runs
   without depending on the custom image at all.

Three flags were each required (every one found by a failed smoke — keep them):
- dockerd must run **`--storage-driver=vfs --iptables=false --bridge=none`**
  (nested overlayfs + bridge/iptables setup fail in a container; sandbox
  containers use `--network none` so no bridge is needed). Else: "dockerd not
  ready".
- runsc runtimeArgs must include **`--ignore-cgroups`** (plus `--platform=ptrace`
  — no /dev/kvm in iris tasks — and `--network=sandbox`). The task cgroup is
  restricted so runsc can't write `cgroup.subtree_control`. Else: "cannot set up
  cgroup for root ... operation not supported".
- sandbox images need **bash** (`GvisorContainerSandbox` execs `bash -lc`).
  `alpine` is busybox-only; TB task images ship bash. Use `debian:stable-slim`
  for any bash smoke.

Build/push the custom image (only when the Dockerfile changes):
```bash
gh auth token | docker login ghcr.io -u <user> --password-stdin
docker buildx build --platform linux/amd64 -f docker/Dockerfile.agent-task \
  -t ghcr.io/<owner>/openthoughts-agent-task:latest --push docker/
# new GHCR packages are private; make it public so the AR mirror can pull it.
```

## Running on iris

`uv run iris --cluster=marin job run ...` from this dir (depends on `marin-iris`).
Cluster config is the packaged `--cluster=marin` lookup (no local file). Auth in
`~/.iris/`; `iris login` if missing.

**SFT (v6e-16):**
```bash
uv run iris --cluster=marin job run --no-wait \
  --tpu v6e-16 --enable-extra-resources --extra tpu \
  --region europe-west4 --region us-east1 --region us-east5 \
  --cpu 24 --memory 400GB --disk 200GB --max-retries 3 --job-name ota-sft-qwen3-8b \
  -e HF_TOKEN "$HF_TOKEN" \
  -e CKPT_DIR gs://marin-us-central2/openthoughts-agent/qwen3-8b-agent-sft \
  -- python launch_sft.py
```

**Eval (v6e-8 + the custom image OR the stock image):**
```bash
uv run iris --cluster=marin job run --no-wait \
  --task-image ghcr.io/rjpower/openthoughts-agent-task:latest \
  --tpu v6e-8 --enable-extra-resources --extra tpu --region europe-west4 \
  --cpu 16 --memory 200GB --disk 200GB --max-retries 1 --job-name ota-eval \
  -e HF_TOKEN "$HF_TOKEN" \
  -e CKPT_DIR gs://marin-us-central2/openthoughts-agent/qwen3-8b-agent-sft \
  -e TASK_LIMIT 10 -- python launch_eval.py
# Drop --task-image to run on the stock image (runtime-bootstraps docker+runsc).
```

**Live gVisor smoke (no checkpoint needed, smallest slice):**
```bash
uv run iris --cluster=marin job run --no-wait \
  --tpu v6e-4 --enable-extra-resources --extra tpu --region europe-west4 \
  --cpu 8 --memory 60GB --disk 60GB --max-retries 1 --job-name ota-gvisor-smoke \
  -- python -m eval.gvisor_smoke
```

### Job knobs (env)
SFT: `AGENT_MODEL SFT_STEPS BATCH_SIZE LR MAX_SEQ_LEN SEED TP DATA_LIMIT
CKPT_DIR REMAT FLASH EVAL_GEN` — see `launch_sft.py` docstring.
Eval: `AGENT_MODEL CKPT_DIR TASK_LIMIT MAX_TURNS COMMAND_TIMEOUT MAX_NEW_TOKENS
TP MAX_PROMPT_LEN TEMPERATURE OTA_SANDBOX` — see `launch_eval.py`.
Sandbox: `OTA_SANDBOX` (gvisor|local), `DOCKERD_ARGS` (override dockerd flags).

## Hard-won invariants (carry these or re-break them)

- **`jax.distributed.initialize()` before any jax call on multi-host.** v6e-8/-16
  span 2/4 hosts; orbax checkpoint barriers crash with "Distributed system is not
  available" without it. `init_distributed()` is the first line of each
  `main()`. (A single-host smoke never hits this — the trap is that it "works"
  small and dies at scale.)
- **Commit before submit.** iris bundles via `git-ls-files`; uncommitted changes
  don't ship. Watch the bundle-size cap — never let local model snapshots /
  checkpoints get tracked (they're gitignored: `*/qwen3-*/`, `*/checkpoints/`).
  A `git add -A` that starts hashing a 3GB weights dir will hang.
- **`--region` is mandatory and v6e is only in 3 zones** (`europe-west4`,
  `us-east1`, `us-east5`). Pass all three for a v6e-16 or it sits pending (4
  hosts is hard to co-schedule in one region).
- **`launch_sft.py` runs as `python launch_sft.py`** (root entrypoint).
  `gvisor_smoke` runs as `python -m eval.gvisor_smoke` (it's a package module).
- **fp32 actor params for SFT.** A 1e-5–8e-5 AdamW update is below bf16 ULP for
  unit-scale weights → bf16 storage silently zeroes updates. Compute in bf16.
- **Gradient clipping is load-bearing.** Unclipped → inf/NaN grad → libtpu
  SIGSEGV that kills the run. Always `clipped_adamw`.
- **tunix `Sampler` decodes GREEDILY unless `top_p` is passed** (ignores
  `temperature` AND `seed`). Pass `top_p=1.0` to every eval/rollout sampler.
- **Generation conflicts with remat.** The tunix Sampler mutates KV-cache Params,
  which trips remat's trace level ("Cannot mutate Param from a different trace
  level"). Train with remat; load with remat=NONE to sample. `EVAL_GEN=0` in SFT.
- **`BATCH_SIZE` must be divisible by the fsdp axis** (`device_count // TP`).
- **Flash/splash attention is OFF by default** — tunix's splash kernel
  shard-maps the batch over fsdp and breaks for small batches. `FLASH=0`.
- **`kv_cache_size >= max_prompt_length + max_new_tokens`** or the sampler errors.
- **Re-`uv lock` before submit if stale** — `marin-*` are nightly `0.2.x.dev`.
- **Host RAM was often the limiter.** `--memory 400GB` for 8B; the sandbox's vfs
  storage driver is disk-hungry — give eval jobs `--disk 200GB`.

## Dependencies
`google-tunix` (0.1.7 PyPI; `qwen3_8b` preset, `PeftTrainer`+orbax, Dr.GRPO),
`marin-iris`, `marin-fray`. `--extra tpu` pulls `jax[tpu]`/libtpu on the worker.
Async-RL work may later switch to the `rjpower/tunix` fork (`~/code/tunix`).

## Local checks (CPU)
```bash
OTA_SANDBOX=local JAX_PLATFORMS=cpu uv run pytest -q   # encoder/masking + agent-loop tests
```
The sandbox/gVisor path can't be tested on CPU (no privilege/docker) — validate it
with `eval/gvisor_smoke.py` on a v6e-4.
