# openthoughts-agent

Post-training **Qwen3-8B** into an **OpenThoughts terminal agent**
(https://www.openthoughts.ai/blog/agent) with `google-tunix`, run on the
**marin/iris** TPU cluster (v6e-8 / v6e-16).

The OpenThoughts agent reads a task + live terminal output and emits structured
shell actions; it's evaluated on **Terminal-Bench** (the `OpenThoughts-TB-dev`
tasks are Docker environments + graders), so tool execution runs in an isolated
**gvisor (runsc)** sandbox.

## Pipeline

| Stage | What | Status |
| --- | --- | --- |
| **1. SFT** | Qwen3-8B on `open-thoughts/OpenThoughts-Agent-v1-SFT` (~15.2k Terminus-2 traces), ChatML + assistant-turn loss masking → orbax checkpoint | done (`launch_sft.py`); deep multi-epoch run training |
| **2. Eval** | SFT'd agent on `open-thoughts/OpenThoughts-TB-dev` / Terminal-Bench, tools in a gvisor sandbox | done + validated (`launch_eval.py`); see `REPORT.md` |
| **3. RL** | Dr.GRPO (tunix agentic) — multi-turn rollouts in the gvisor sandbox, sparse grader reward | built (`launch_rl.py`); smoke in progress |

## Quick start

```bash
# CPU: fast unit tests (ChatML masking; tokenizer only, no weights)
uv run pytest -q

# TPU SFT on iris (8B): see AGENTS.md for the full submit recipe
uv run iris --cluster=marin job run --no-wait \
  --tpu v6e-16 --enable-extra-resources --extra tpu --region europe-west4 \
  --cpu 8 --memory 200GB --disk 200GB --max-retries 1 --job-name ota-sft-qwen3-8b \
  -e HF_TOKEN "$HF_TOKEN" -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e CKPT_DIR gs://marin-us-central2/openthoughts-agent/qwen3-8b-sft \
  -- python launch_sft.py
```

See **`AGENTS.md`** for the cluster recipe, env knobs, and the hard-won
invariants (fp32 actor, gradient clipping, the greedy-sampler `top_p` fix,
commit-before-submit, etc.).

## Layout

```
launch_sft.py            # iris entrypoint (stage 1: SFT)
launch_eval.py           # iris entrypoint (stage 2: Terminal-Bench eval)
launch_rl.py             # iris entrypoint (stage 3: Dr.GRPO RL)
models/                  # HF Qwen3 -> tunix nnx loader, registry, orbax checkpoints
agent_data/agent_traces.py  # stream OpenThoughts-Agent-v1-SFT (pinned revision)
training/agent_sft.py    # ChatML encoder + assistant masking + PeftTrainer + orbax ckpt
training/common.py       # clipped_adamw, build_mesh(tp), metrics_logging, init_distributed
eval/                    # gvisor sandbox + Terminal-Bench harness (agent loop, grade)
rl/                      # Dr.GRPO env + agent (tunix.rl.agentic) over the gvisor sandbox
tests/                   # CPU tests (encoder/masking, agent loop, RL env)
```

See **`AGENTS.md`** → "Common tasks" for copy-paste submit commands for every stage.

## Provenance

Built on the `tunix-delphi-rl` feasibility study
([marin-experiments PR #2](https://github.com/rjpower/marin-experiments/pull/2)):
the tunix Qwen3 loader, the FSDP mesh, `clipped_adamw`, and the loss-masked
`PeftTrainer` SFT pattern are adapted from it. New here: the ChatML agent-trace
encoder, the pyarrow trace loader, orbax checkpointing, and the 8B memory wiring
(fp32 params + decoder remat + flash attention + optional tensor parallelism).
