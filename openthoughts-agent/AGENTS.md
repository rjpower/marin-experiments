# openthoughts-agent — agent notes

Goal: build the best **OpenThoughts terminal agent** on **Qwen3-8B**
(https://www.openthoughts.ai/blog/agent), trained with `google-tunix` and run on
the **marin/iris** TPU cluster (v6e-8 / v6e-16 slices). Read the repo-root
`/AGENTS.md` (shared practices) first; only experiment-specific conventions are here.

**New here? Read this top-to-bottom once, then jump to "Common tasks".** The three
stages each have one root entrypoint: `launch_sft.py`, `launch_eval.py`,
`launch_rl.py`. Everything else is a library they call.

## Status (2026-06-21)

| Stage | What | State |
|------|------|------|
| 1. SFT | Qwen3-8B on `OpenThoughts-Agent-v1-SFT` (~15.2k Terminus-2 traces) → GCS ckpt | **done**; baseline ckpt at `…/qwen3-8b-agent-sft`; 3-epoch **deep** run complete (11.4k steps, loss 0.043) at `…/qwen3-8b-agent-sft-deep` |
| 2. Eval | SFT'd agent on `OpenThoughts-TB-dev` (Terminal-Bench) in a gVisor sandbox | **done + validated**: clean end-to-end, oracle grades 1.0, baseline solved 0/5 (format learned, capability low) |
| 3. RL | Dr.GRPO (tunix agentic), sync rollouts | **built + validated at 8B**: full loop ran end-to-end single-host (1.7B/v6e-4 TP=4), multi-host (1.7B/v6e-8), AND **8B on v6e-16 TP=8** (`ota-rl-8b-fit2`, restore→rollout→grade→train_step); real run gated on pass@k spread from the deep-ckpt gate eval |

Milestone-1 write-up: `REPORT.md`. PR: rjpower/marin-experiments#9.

## Architecture (3 stages, one entrypoint each)

```
            OpenThoughts-Agent-v1-SFT traces
                          │  launch_sft.py
                          ▼
   Qwen3-8B  ──SFT (ChatML, assistant-turn loss mask)──▶  orbax ckpt (gs://)
                          │
        ┌─────────────────┴───────────────────┐
        │ launch_eval.py                       │ launch_rl.py
        ▼                                      ▼
  per TB task: build image → gVisor      per task × G gens: rollout the agent
  sandbox → run agent loop → grade       in the gVisor sandbox → grade = reward
  (pass@1)                               → Dr.GRPO update (advantage = group spread)
```

The **agent loop** is one shape everywhere (eval and RL): the policy emits a
Terminus-2 JSON action `{analysis, plan, commands:[{keystrokes,…}], task_complete}`;
we exec the commands in the sandbox and feed the terminal output back as the next
user turn. RL reuses the eval loop's `parse_action` / `format_observation` /
`SYSTEM_PROMPT` so rollouts stay in the SFT distribution.

## Source map

**Entrypoints (root):**
- `launch_sft.py` — stage 1. Loads Qwen3, SFTs on agent traces, orbax-checkpoints.
- `launch_eval.py` — stage 2. Per TB task: build → sandbox → `run_episode` → grade.
- `launch_rl.py` — stage 3. RLCluster + Dr.GRPO over the agentic env/agent.

**`models/`** — model loading + checkpoints.
- `qwen3_loader.py` — stock HF Qwen3 → tunix `flax.nnx` Qwen3. Data-driven from
  `config.json`; threads `param_dtype` (fp32 SFT), `remat`, `use_flash_attention`.
- `registry.py` — name → repo + loaders: `qwen3-8b` (agent base), `qwen3-8b-base`,
  `qwen3-1.7b-base` (fast smoke).
- `checkpoint.py` — `restore_sft_model(base_dir, ckpt_dir, mesh)`: load base
  (remat=NONE so the sampler works) + orbax `maybe_restore`.

**`agent_data/`** — SFT data.
- `agent_traces.py` — stream `OpenThoughts-Agent-v1-SFT` (pinned rev) via
  `snapshot_download` + **pyarrow** (datasets<4 can't parse the saved `List` type).
  Dir is `agent_data/`, NOT `data/` (repo `.gitignore` drops `data/` from the bundle).

**`training/`** — training glue (shared by SFT + RL).
- `agent_sft.py` — ChatML encoder (`render_chatml`, `encode_agent_conversation`
  with assistant-content+`<|im_end|>` loss mask), grain source, `run_agent_sft`.
- `common.py` — `init_distributed`, `clipped_adamw`, `build_mesh(tp)`,
  `sft_model_input_fn`, `metrics_logging_options` (opt-in wandb/tb).

**`eval/`** — the sandbox + Terminal-Bench harness (the sandbox is shared by RL).
- `sandbox.py` — **the sandbox layer**: `ensure_sandbox_runtime`, `ensure_dockerd`,
  `build_image`, `GvisorContainerSandbox`, `LocalUnsafeSandbox`, `make_sandbox`.
- `agent_loop.py` — `SYSTEM_PROMPT`, `parse_action` (last balanced JSON w/ `commands`),
  `format_observation`, `run_episode`.
- `tb_tasks.py` — load `OpenThoughts-TB-dev` tasks (`TBTask`: env dir, tests, timeouts).
- `grade.py` — copy tests in, run `tests/test.sh`, parse `reward.txt`.
- `model_serving.py` — SFT'd model → `model_fn(messages)->str` (tunix Sampler, `top_p=1.0`,
  context trimmed to `max_prompt_length`).
- `gvisor_smoke.py` — live gVisor validation (no model). `eval_smoke.py` — model-free
  oracle-solve grading smoke.

**`rl/`** — Dr.GRPO via tunix's *agentic* learner (`tunix.rl.agentic`).
- `agent.py` — `TerminusAgent(ConversationAgentBase)`: parses the JSON action,
  no system turn (matches the SFT traces).
- `environment.py` — `TerminalBenchEnv(BaseTaskEnv)`: boots the task image under
  gVisor, execs the agent's shell per step, grades at episode end (sparse reward).
  `register_tasks(...)` populates the id→TBTask registry the envs look up.

**`docker/Dockerfile.agent-task`** — custom iris task image (runsc + docker + buildx).
**`tests/`** — CPU tests (encoder/masking, agent loop, RL env+agent).

## Common tasks

All `uv run …` commands run from this directory.

**Run the CPU tests** (fast; encoder/masking, agent loop, RL env — no weights/TPU):
```bash
OTA_SANDBOX=local JAX_PLATFORMS=cpu uv run pytest -q
```

**Run a quick offline check of an entrypoint** (imports + config construction; catches
API drift without a TPU):
```bash
JAX_PLATFORMS=cpu uv run python -c "import launch_rl, launch_sft, launch_eval; print('ok')"
```

**Submit an SFT run** (v6e-16; flash + TP=4 + batch 4 fits 8B at seq 8192). For a
**deep** (multi-epoch) run, raise `SFT_STEPS` (≈3800 steps/epoch at batch 4) and set
`WANDB_PROJECT` for loss curves:
```bash
uv run iris --cluster=marin job run --no-wait \
  --tpu v6e-16 --enable-extra-resources --extra tpu \
  --region europe-west4 --region us-east1 --region us-east5 \
  --cpu 24 --memory 400GB --disk 200GB --max-retries 3 --job-name ota-sft-deep \
  -e HF_TOKEN "$HF_TOKEN" -e WANDB_API_KEY "$WANDB_API_KEY" -e WANDB_PROJECT openthoughts-agent \
  -e AGENT_MODEL qwen3-8b -e SFT_STEPS 11400 -e BATCH_SIZE 4 -e TP 4 -e FLASH 1 \
  -e CKPT_DIR gs://marin-us-central2/openthoughts-agent/qwen3-8b-agent-sft-deep \
  -- python launch_sft.py
```
`PeftTrainer.maybe_restore` resumes params+step from `CKPT_DIR`, so a preempted long
run continues on retry (point a fresh run at a NEW dir to train from base).

**Submit an eval** (v6e-8; stock image runtime-bootstraps runsc, or add `--task-image`
for the baked image):
```bash
uv run iris --cluster=marin job run --no-wait \
  --tpu v6e-8 --enable-extra-resources --extra tpu \
  --region europe-west4 --region us-east1 --region us-east5 \
  --cpu 16 --memory 200GB --disk 250GB --max-retries 1 --job-name ota-eval \
  -e HF_TOKEN "$HF_TOKEN" -e CKPT_DIR gs://.../qwen3-8b-agent-sft \
  -e TASK_LIMIT 5 -- python launch_eval.py
```
**pass@k eval** (the RL gate — find tasks with reward spread): add `-e K_SAMPLES 8
-e TEMPERATURE 0.8`. It runs each task k times and prints `pass@1`, `pass@k`, per-task
`score[min/mean/max]`, and the `RL-TRAINABLE` task list. The RL gate is **`score_spread`**
(the CONTINUOUS grader score varies across the k samples), NOT binary `0<pass1<1`: the RL
env reward is the continuous score (`rl/environment.py`), so Dr.GRPO gets advantage from
score variance even with 0 full solves. Greedy (temp ~0) gives no spread — need temperature.
Cost ≈ k × the pass@1 sweep.

**Fan the eval out (v6e-8s are plentiful).** Eval is sequential within a process (~7 min/
task at 8B, k=5, 10 turns) so a 70-task sweep crawls. Shard with `TASK_OFFSET`+`TASK_LIMIT`
across many jobs — finest is one task per job (`TASK_LIMIT 1 TASK_OFFSET <i>`, `--disk 50GB`).
Each task frees its image after grading (`remove_image`), so 50GB suffices. **zsh gotcha:**
the Bash tool runs zsh, which does NOT word-split unquoted `$var` — `for i in $LIST` runs
ONCE with the whole string (silently makes one job named `…-1 2 3 …`). Use a literal list:
`for i in 11 12 13 …; do`.

**Submit an RL run.** Single-host **machinery smoke** first (validates the rollout →
gVisor → grade → Dr.GRPO loop without multi-host complexity):
```bash
uv run iris --cluster=marin job run --no-wait \
  --tpu v6e-4 --enable-extra-resources --extra tpu \
  --region europe-west4 --region us-east1 --region us-east5 \
  --cpu 8 --memory 120GB --disk 200GB --max-retries 1 --job-name ota-rl-smoke \
  -e HF_TOKEN "$HF_TOKEN" -e AGENT_MODEL qwen3-1.7b-base \
  -e TASK_LIMIT 2 -e NUM_GENERATIONS 4 -e PROMPTS_PER_BATCH 1 -e RL_STEPS 2 \
  -e MAX_TURNS 5 -e MAX_CONCURRENCY 4 -- python launch_rl.py
```
**Real 8B run** (validated config — `ota-rl-8b-fit2` ran this end-to-end): `--tpu
v6e-16`, `-e AGENT_MODEL qwen3-8b -e TP 8`, `-e CKPT_DIR` = the (deep) SFT ckpt,
`-e RL_CKPT_DIR` set to save, `--disk 100GB`, `WANDB_PROJECT` set. Start near the fit
envelope (`MAX_PROMPT_LEN 4096 MAX_RESPONSE_TOKENS 768 MAX_TURNS 3`); larger G / longer
episodes may need TP=16 or shorter seqs. **For RL to learn, the SFT policy must solve
the chosen tasks *sometimes* (pass@k > pass@1 > 0)** — otherwise every generation
scores 0, advantages are 0, and nothing updates (the bimodal wall; see REPORT.md). Use
the **pass@k eval above** to pick those tasks (`K_SAMPLES`/`TEMPERATURE`).

**Live gVisor smoke** (no model, smallest slice — proves isolation works):
```bash
uv run iris --cluster=marin job run --no-wait \
  --tpu v6e-4 --enable-extra-resources --extra tpu --region europe-west4 \
  --cpu 8 --memory 60GB --disk 60GB --max-retries 1 --job-name ota-gvisor-smoke \
  -- python -m eval.gvisor_smoke
```

**Watch / inspect a job:**
```bash
uv run iris --cluster=marin job summary /power/<job-name>          # state + per-task
uv run iris --cluster=marin job logs /power/<job-name> --max-lines 5000 | grep '\[ota-'
```

**Build + push the custom task image** (only when `docker/Dockerfile.agent-task` changes;
new GHCR packages are private → make it public so the AR mirror can pull it):
```bash
gh auth token | docker login ghcr.io -u <user> --password-stdin
docker buildx build --platform linux/amd64 -f docker/Dockerfile.agent-task \
  -t ghcr.io/rjpower/openthoughts-agent-task:latest --push docker/
```

### Job knobs (env)
- **SFT** (`launch_sft.py`): `AGENT_MODEL SFT_STEPS BATCH_SIZE LR MAX_SEQ_LEN SEED TP
  DATA_LIMIT CKPT_DIR REMAT FLASH EVAL_GEN RUN_NAME WANDB_PROJECT`.
- **Eval** (`launch_eval.py`): `AGENT_MODEL CKPT_DIR TASK_LIMIT MAX_TURNS COMMAND_TIMEOUT
  MAX_NEW_TOKENS TP MAX_PROMPT_LEN TEMPERATURE K_SAMPLES OTA_SANDBOX`.
- **RL** (`launch_rl.py`): `AGENT_MODEL CKPT_DIR RL_CKPT_DIR RL_STEPS NUM_GENERATIONS
  PROMPTS_PER_BATCH TASK_LIMIT MAX_TURNS MAX_RESPONSE_TOKENS MAX_PROMPT_LEN TEMPERATURE LR
  BETA TP MAX_CONCURRENCY COMMAND_TIMEOUT EPISODE_TIMEOUT RUN_NAME WANDB_PROJECT`.
- **Sandbox**: `OTA_SANDBOX` (gvisor|local), `DOCKERD_ARGS` (override dockerd flags).
- **Metrics**: set `WANDB_PROJECT` (+ `WANDB_API_KEY`, inherited by the job) for wandb;
  `TB_LOG_DIR` for tensorboard. Unset → stdout only.

## gVisor sandbox (the agent runs untrusted shell — isolate it)

Each TB task is its own Docker environment; we run it under **runsc** so the agent's
commands hit gVisor's emulated kernel, not the host. **CONFIRMED working** inside an
iris TPU task: a `--runtime=runsc` container reports kernel `4.19.0-gvisor` while the
host is `6.8.0-gcp`. This works because **TPU tasks run `--privileged`** (iris adds it
for accelerators); a **CPU-only** iris task is NOT privileged, so the sandbox needs a
`--tpu` slice.

`ensure_sandbox_runtime()` handles both delivery paths: the custom `--task-image` bakes
runsc+docker in (bootstrap is a no-op), or on the stock iris image it downloads the
static binaries at runtime. Three flags were each required (every one found by a failed
smoke — keep them):
- dockerd: **`--storage-driver=vfs --iptables=false --bridge=none`** (nested overlayfs +
  bridge/iptables setup fail in a container; sandbox containers use `--network none`).
- runsc runtimeArgs: **`--ignore-cgroups`** (restricted task cgroup), `--platform=ptrace`
  (no /dev/kvm), `--network=sandbox`.
- builds use **`--network=host`** (dockerd is bridgeless) so apt has egress.
- sandbox images need **bash** (`GvisorContainerSandbox` execs `bash -lc`); use
  `debian:stable-slim` for any bash smoke (alpine is busybox-only).

## Hard-won invariants (carry these or re-break them)

- **`jax.distributed.initialize()` before any jax call on multi-host.** v6e-8/-16 span
  2/4 hosts; orbax barriers crash without it. `init_distributed()` is line 1 of each
  `main()`. (A single-host smoke never hits this — it "works" small and dies at scale.)
- **Commit before submit.** iris bundles via `git-ls-files`; uncommitted changes don't
  ship. Model snapshots / checkpoints are gitignored (`*/qwen3-*/`, `*/checkpoints/`);
  a `git add -A` that starts hashing a 3GB weights dir will hang.
- **`--region` is mandatory; v6e is only in 3 zones** (`europe-west4`, `us-east1`,
  `us-east5`). Pass all three for a v6e-16 or it sits pending.
- **fp32 actor params for training.** A 1e-5 AdamW update is below bf16 ULP for
  unit-scale weights → bf16 storage silently zeroes updates. Compute in bf16.
- **Gradient clipping is load-bearing.** Unclipped → inf/NaN grad → libtpu SIGSEGV.
  Always `clipped_adamw`.
- **tunix `Sampler` decodes GREEDILY unless `top_p` is passed** (ignores temperature
  AND seed). Pass `top_p=1.0` to every eval/rollout sampler.
- **Generation conflicts with remat.** The Sampler mutates KV-cache Params, tripping
  remat's trace level. Train with remat; load with **remat=NONE** to sample. So the
  SFT job uses `EVAL_GEN=0`, and eval/RL load the actor with remat=NONE.
- **`BATCH_SIZE` must be divisible by the fsdp axis** (`device_count // TP`).
- **8B memory ladder (SFT):** flash + TP=4 + batch 4 fits seq 8192 (70.6G→33.4G→fits).
- **Agent context grows every turn** (terminal output); trim the prompt to
  `max_prompt_length` (`model_serving._fit_messages`) or the sampler KV cache overflows.
- **RL needs reward spread.** Dr.GRPO advantage = reward − group mean; an all-0 (or
  all-1) group yields 0 advantage. Train on tasks the SFT policy solves *sometimes*.
- **RL rollouts run sandbox containers in-process (sync).** Validated single-host
  (1.7B/v6e-4, TP=4) AND multi-host (1.7B/v6e-8, 2 hosts) — both ran the full
  rollout→gVisor→grade→Dr.GRPO loop to completion (`ota-rl-smoke6`, `ota-rl-mh`).
- **Multi-host agentic RL works** (despite no `process_index` guards in the tunix
  learner): both the 1.7B/v6e-8 and the **8B/v6e-16** runs completed cleanly, no
  collective desync/hang. The 8B run is unblocked on that axis.
- **RL backprop OOMs without sharding** at `remat=NONE` (forced by the sampler). 1.7B
  OOMs v6e-4 at TP=1 (42G temporaries); **TP shards the per-sequence activations** →
  TP=4 fits 1.7B on v6e-4. **8B fits v6e-16 at TP=8** (`ota-rl-8b-fit2`: restore from
  SFT ckpt → rollout G=2 → grade → train_step, at prompt 4096 / response 768 / 3 turns,
  with headroom). Larger G / longer episodes may need TP=16 or shorter seqs.
- **8B RL timing** (v6e-16, the fit probe): restore ~11 min, image build ~2 min, then
  ~4 min for 1 step at G=2 / 3 turns. Per-step cost scales with G × turns × tokens.
- **Disk: request ≤ a single node's free space.** `--disk 300GB` was rejected
  (autoscaler `insufficient_resources: disk`); 100GB ran the 8B RL probe fine (model +
  a few task images under vfs). Multi-task eval has needed 200–250GB (vfs doesn't share
  image layers, so disk ≈ Σ task-image sizes). Keep RL disk modest (100GB).
- **Re-`uv lock` before submit if stale** — `marin-*` are nightly `0.2.x.dev`.

## Dependencies
`google-tunix` (0.1.7; `qwen3_8b` preset, `PeftTrainer`+orbax, `tunix.rl.agentic`
Dr.GRPO), `marin-iris`, `marin-fray`, `wandb` (metrics). `--extra tpu` pulls
`jax[tpu]`/libtpu on the worker. Async-RL work may later switch to the `rjpower/tunix`
fork (`~/code/tunix`).

## Local checks (CPU)
```bash
OTA_SANDBOX=local JAX_PLATFORMS=cpu uv run pytest -q   # encoder/masking, agent loop, RL env
```
The sandbox/gVisor path can't run on CPU (no privilege/docker) — validate it with
`eval/gvisor_smoke.py` on a v6e-4.
