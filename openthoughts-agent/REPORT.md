# OpenThoughts agent on Qwen3-8B — Milestone 1 report

**Goal (weaver #240):** build the best OpenThoughts terminal agent on Qwen3-8B
(https://www.openthoughts.ai/blog/agent), trained with `google-tunix` on the
marin/iris TPU cluster (v6e-8 / v6e-16). **Milestone 1:** get a basic SFT
experiment running and evaluated against the OpenThoughts benchmark, with agent
tool execution isolated in **gVisor** containers launched from inside the iris
TPU task.

## Status: milestone 1 complete

The full pipeline runs end-to-end on the cluster:

```
OpenThoughts-Agent-v1-SFT traces ──SFT──▶ Qwen3-8B checkpoint (GCS)
                                            │
OpenThoughts-TB-dev tasks ──build──▶ gVisor sandbox ──agent loop──▶ grade
```

| Component | Result |
|---|---|
| **gVisor sandbox** on a privileged iris TPU task | ✅ confirmed live: containers run gVisor kernel `4.19.0-gvisor` (host `6.8.0-gcp`); exec + file copy verified |
| **Qwen3-8B SFT** on v6e-16 | ✅ 1000 steps, ChatML + assistant-turn loss mask, orbax checkpoint at `gs://marin-us-central2/openthoughts-agent/qwen3-8b-agent-sft/1000/` |
| **Eval harness** (Terminal-Bench) | ✅ validated end-to-end: a TB task built under gVisor + its oracle solution graded to **score 1.0** |
| **First agent eval** (Qwen3-8B-SFT) | runs clean (0 infra errors), 20-turn loops, **format learned** (parse-failures mostly 0), **solved 0/5** then **0/20** on a wider sample (capability floor, not a harness artifact) |

The 0/5 is an honest capability baseline, not a harness artifact: the same
harness scores a task's oracle solution 1.0, and the SFT model emits well-formed
Terminus-2 JSON actions (3 of 5 tasks had **zero** parse failures across 20
turns). The model is only ~0.6 epoch of SFT (effective batch 4 × 1000 steps) and
Terminal-Bench is hard; raising the pass rate is exactly what milestones 2-3
(more SFT / data-gen, then Dr.GRPO RL) are for.

## What was built

`openthoughts-agent/` (on top of rjpower/marin-experiments PR#2):

- **Model loading** (`models/`): stock HF Qwen3 → tunix native `flax.nnx` Qwen3,
  data-driven from `config.json` (fp32 params for SFT, bf16 compute, decoder
  remat, optional flash attention); orbax checkpoint save/restore (restore
  reshards across a different TP than training).
- **SFT** (`training/`, `agent_data/`): stream `OpenThoughts-Agent-v1-SFT` traces
  (parquet via pyarrow — `datasets<4` can't parse the saved `List` feature),
  encode in real Qwen3 ChatML with assistant-turn loss masking, train with tunix
  `PeftTrainer` + global-norm-clipped AdamW.
- **gVisor sandbox** (`eval/sandbox.py`): run each TB task image under the
  `runsc` OCI runtime; `ensure_sandbox_runtime()` works whether the binaries are
  baked into the custom task image OR bootstrapped at runtime on the stock image.
- **Agent loop** (`eval/agent_loop.py`): the Terminus-2 JSON-action loop
  (`{analysis, plan, commands}`), with a robust last-balanced-JSON-object parser
  and context trimming to the sampler's prompt budget.
- **Eval** (`eval/tb_tasks.py`, `eval/grade.py`, `launch_eval.py`): load TB-dev
  tasks, build each image, drive the policy in the sandbox, run the task's
  `tests/test.sh`, read `reward.txt`.
- **Custom task image** (`docker/Dockerfile.agent-task`,
  `ghcr.io/rjpower/openthoughts-agent-task`): iris-task base + docker + runsc +
  buildx; selected per-job with `--task-image`.

## Engineering findings (the hard parts)

**1. gVisor inside an iris TPU task.** TPU tasks run `--privileged` (iris adds it
for accelerators) → rootful gVisor + a task-local dockerd work. CPU-only iris
tasks are *not* privileged, so the sandbox needs a `--tpu` slice. Three flags
were each required, found via successive failed smokes:
- dockerd nested: `--storage-driver=vfs --iptables=false --bridge=none` (nested
  overlayfs + bridge/iptables setup otherwise hang dockerd at startup);
- runsc: `--ignore-cgroups` (the task cgroup is restricted, so runsc can't write
  `cgroup.subtree_control`), plus `--platform=ptrace` (no /dev/kvm) and
  `--network=sandbox`;
- builds use `--network=host` (dockerd is bridgeless) so `apt` has egress.

**2. The model-free eval smoke earned its keep.** Before spending the 8B
checkpoint on the eval, a model-free smoke (`eval/eval_smoke.py`: build a real TB
image under gVisor, run its *oracle* solution, grade) surfaced **five** distinct,
real bugs for ~5 min of v6e-4 each instead of one-at-a-time mid-eval:
Docker 27 dropped the legacy builder (→ install buildx) · HF stores task files as
blob symlinks BuildKit can't follow (→ dereference the build context) · bridgeless
dockerd (→ `--network=host`) · single-file `docker cp` lands broken under runsc (→
copy into the parent dir) · ~half of TB-dev oracles are stubs (→ select real ones).

**3. Fitting 8B SFT on v6e-16.** 8B (fp32 params + AdamW) at seq 8192 OOMs the
31.25 GB/chip HBM. The temporaries ladder: **70.6 GB** (TP=1, bs16) → **33.4 GB**
(flash + TP=2, bs8) → **fits** (~20-22 GB) at **flash + TP=4 + batch 4**. Levers:
model/optimizer/grad states are fixed (~8 GB, sharded /16 regardless of fsdp×tp
split); flash kills the seq² attention term; raising TP shards the model-dim
temporaries (batch is already 1 seq/device, so lowering it doesn't help).

**4. Multi-host gotchas.** (a) `jax.distributed.initialize()` must run before any
jax call or orbax checkpoint barriers crash with "Distributed system is not
available" — a single-host smoke never hits this, so it only bites at scale.
(b) v6e lives in only 3 zones (europe-west4, us-east1, us-east5); a v6e-16 (4
hosts) sits pending unless all three `--region`s are passed.

**5. Agent context management.** Terminal output accumulates every turn; the
tunix Sampler uses the *actual* prompt length, so an unbounded agent context
blows the KV cache. Fix: trim history to `max_prompt_length`, keeping the system
+ task message and the most recent turns.

## How to run

See `AGENTS.md` for full recipes. In short:

```bash
# SFT (v6e-16): flash + TP=4 + batch 4 fits 8B at seq 8192
uv run iris --cluster=marin job run --tpu v6e-16 --enable-extra-resources --extra tpu \
  --region europe-west4 --region us-east1 --region us-east5 \
  -e CKPT_DIR gs://.../qwen3-8b-agent-sft -e FLASH 1 -e TP 4 -e BATCH_SIZE 4 \
  -- python launch_sft.py

# Eval (v6e-8): stock image bootstraps runsc, or add --task-image for the baked one
uv run iris --cluster=marin job run --tpu v6e-8 --enable-extra-resources --extra tpu \
  -e CKPT_DIR gs://.../qwen3-8b-agent-sft -e TASK_LIMIT 5 -- python launch_eval.py
```

## Phase 2 (in progress): deep SFT + RL

Built on top of milestone 1 (same branch / PR):

- **Metrics logging** — opt-in wandb/tensorboard wired through both SFT and RL
  (`training.common.metrics_logging_options`, gated on `WANDB_PROJECT`). The
  `PeftTrainer` prints no loss to stdout, so this is the only training signal;
  validated live on the deep-SFT job.
- **Deep SFT** — a 3-epoch run (≈11.4k steps at batch 4) from base into a fresh
  checkpoint dir, wandb-tracked, resumable via `PeftTrainer.maybe_restore`. The
  milestone-1 baseline was only ~0.26 epoch.
- **RL stage (Dr.GRPO)** — multi-turn RL via tunix's *agentic* learner
  (`tunix.rl.agentic`), so the gVisor harness is the reward environment:
  - `rl/agent.py` `TerminusAgent` parses the Terminus-2 JSON action (reuses the
    eval loop) with no system turn — matching the SFT traces' role layout.
  - `rl/environment.py` `TerminalBenchEnv` boots the task image under gVisor, execs
    the agent's shell per turn, and grades the container at episode end → sparse
    reward. With `reward_fns=None` the agentic reward manager uses this env reward;
    `advantage_estimator="drgrpo"` makes the per-group reward spread the advantage.
  - `launch_rl.py` wires the RLCluster (vanilla rollout — no vLLM), Dr.GRPO config,
    and `QwenChatTemplateParser(enable_thinking=True)` (tokenization verified
    byte-identical to the SFT encoder).
  - **Validated end-to-end**: the full rollout → 4 concurrent gVisor containers →
    generation → multi-turn env stepping → grading → Dr.GRPO `train_step` loop ran
    to completion on a single host (1.7B/v6e-4, TP=4) **and** across hosts
    (1.7B/v6e-8). Multi-host agentic rollout — the main risk, since the tunix
    learner has no per-host guards — works.

Findings from bringing RL up: tunix enforces `max_tokens_to_generate ==
max_response_length` and `return_logprobs=True`, and `mini_batch_size` is in prompts
(must divide `PROMPTS_PER_BATCH`). A real concurrency bug surfaced — G rollouts boot
containers simultaneously and collided on `pid+ms` names (uuid fix). RL backprop OOMs
at `remat=NONE` (forced by the sampler) unless TP shards the activations (TP=4 fits
1.7B on v6e-4; 1.7B OOMs at TP=1, matching prior experience).

## Next steps

- **Eval the deep-SFT checkpoint** when it lands — the gate for RL is reward spread
  (pass@k > pass@1 > 0); the 0/20 baseline has none, so RL would no-op (bimodal wall).
- **8B RL run** — v6e-16, the deep-SFT ckpt, and the 8B memory-fit ladder (higher TP /
  shorter seqs at `remat=NONE`). Pick RL tasks the SFT policy solves *sometimes*.
- **Larger / generated SFT data**; **async RL rollouts** (may modify the tunix fork).
