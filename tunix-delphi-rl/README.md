# tunix-delphi-rl

Taking a **raw base LM** — no chat template, can't emit JSON, never emits EOS —
and bootstrapping it into an **agentic tool user** with a reusable 3-stage recipe
(SFT for token format → SFT for tool/JSON calling → RL curriculum), on
**[google/tunix](https://github.com/google/tunix)** + **TPUs** via marin's
**[iris](https://github.com/marin-community/marin/tree/main/lib/iris)** cluster.

The model is **Delphi** (`marin-community/delphi-3e18-447Mparams-1.2Btokens`): a
stock **447M Qwen3** base LM with a Llama-3 tokenizer and **no chat template**. We
make it call an external calculator (`CALC(a * b)`) and *chain* the calls.

> **Headline result.** With the recipe below, Delphi does genuine multi-step tool
> use — **single (T0), two-deep (T1), and three-deep chained (T2)** calculator
> calls, where each tool's output is the next call's argument — all reaching
> **~100% solve** on a single **v6e-4** in minutes per stage. The binding
> constraint is never the RL or the iris/TPU plumbing; it's the base LM's
> format/copy priors, and a **distribution-matched SFT warm-up** is exactly the
> lever that fixes them. RL *alone* provably cannot get there.

This started as a feasibility study (*can marin adopt tunix on iris with a marin
model?* — **yes**, see [`REPORT.md`](REPORT.md) §1–§8 and [`DESIGN.md`](DESIGN.md)).
The agentic tool-use work — the focus of this README — is the follow-on
([`REPORT.md`](REPORT.md) §9). For working in this directory as a contributor,
read [`AGENTS.md`](AGENTS.md).

---

## The 3-stage bootstrap recipe

A raw base LM cannot do agentic tool use out of the box: it has no chat template,
emits text not JSON, and never produces EOS. The recipe turns it into a tool user
in three conceptual stages, ordered by what each one teaches:

1. **SFT for token format** — teach the base LM the raw *transcript token format*
   it will operate in: the `Q: ... / CALC(...) / Tool result: ... / answer`
   line structure, with **per-turn loss masking** so it learns to *emit* its own
   turns (the `CALC(...)` calls and the final answer) and *copy* tool results —
   not to produce the environment's lines (`Q:` and `Tool result:` are masked
   out, mask 0).

2. **SFT for tool/JSON calling** — teach the tool-call *surface* and the
   result-*copy* behavior. For this base LM, Qwen JSON tool calls
   (`<tool_call>{…}</tool_call>`) were **out-of-distribution** — it got operands
   mostly right but mangled `op`/braces, so the tool never executed
   (`tool_call_rate ≈ 0` on TPU). So the surface is a **bare-text `CALC(a * b)`**
   call, which the model emits cleanly. Copying an injected `Tool result: X`
   forward (into the next call or the final answer) is also OOD, so RL can't
   bootstrap it — SFT must put it in-distribution first.

3. **RL curriculum** — staged GRPO on increasing chaining depth: **T0** (single
   call `a*b`) → **T1** (two chained calls `a*b*c`) → **T2** (three chained calls
   `a*b*c*d`). Each stage's RL learns to *call* the tool reliably with the right
   operands; the SFT warm-up is what makes the *copy/chain* learnable.

**How the code realizes the stages.** In the current code, stages **1 and 2 are
merged** into a single SFT warm-up (`run_sft_warmup` in `agentic_sft.py`): one
synthetic transcript per stage carries both the line format *and* the
`CALC`/copy surface, trained together with per-turn masking. That is the right
default here (the format and the surface are taught by the same transcript). You
would **split** them if the token format and the tool surface needed to be
learned from different data — e.g. a large generic format/transcript corpus
(stage 1) followed by a small tool-specific corpus (stage 2), or when adding a
second tool whose surface differs. The RL curriculum (stage 3) is
`train_agentic_t0/t1/t2` in `train_agentic.py`, each calling the shared
`_train_agentic_calc` with a per-stage `depth`.

---

## Running it on iris

`launch_agentic.py` is the iris entrypoint. It downloads Delphi from HF at
runtime, applies the worker-shippable RoPE monkeypatch (inside `load_delphi`),
runs the optional SFT warm-up **and** the RL stage on the **same in-memory actor**
(no checkpoint round-trip), and prints per-step metrics. It is configured
entirely by environment variables; `DELPHI_AGENT_MODE` selects the stage:

| `DELPHI_AGENT_MODE` | what runs | tool calls |
|---|---|---|
| `port` (default) | single-turn, **no-tool** arithmetic (the agentic-plumbing smoke test) | 0 |
| `t0` | single calculator call `a * b` | 1 |
| `t1` | two chained calls `a * b * c` | 2 chained |
| `t2` | three chained calls `a * b * c * d` | 3 chained |

| env var | meaning | default |
|---|---|---|
| `DELPHI_AGENT_MODE` | `port` \| `t0` \| `t1` \| `t2` | `port` |
| `DELPHI_STEPS` | GRPO steps | `200` |
| `DELPHI_SFT_STEPS` | SFT warm-up transcripts before RL (`t0/t1/t2` only); **0 skips** | `0` |
| `DELPHI_SFT_LR` | SFT AdamW lr (global-norm-clipped) | `1e-4` |
| `DELPHI_LR` | RL actor lr (**1e-5** learns; 1e-6 too low) | `1e-5` |
| `DELPHI_NUM_GENERATIONS` | GRPO group size | `8` |
| `DELPHI_BATCH_SIZE` | prompts per step | `8` |
| `DELPHI_STAGE` | arithmetic stage (`port` only) | `0` |
| `DELPHI_USE_ROLLOUT_LOGPS` | log the sampler-vs-trainer `logp_diff` canary | `0` |
| `DELPHI_MODEL_DIR` | local dir for the downloaded weights | `./delphi` |

The winning recipe is **prefix-aligned SFT warm-up (~150–250 transcripts,
gradient-clipped) → gradient-clipped GRPO**. Submit a stage on a single-host TPU
(the orchestrator submits; do **not** submit from inside the experiment):

```bash
# T2 (three chained calls): SFT warm-up then RL on one v6e-4
uv run iris --cluster=marin job run --no-wait \
  --tpu v6e-4 --enable-extra-resources --extra tpu --region europe-west4 \
  --cpu 8 --memory 64GB --disk 60GB --max-retries 5 --job-name tunix-t2 \
  -e DELPHI_AGENT_MODE t2 -e DELPHI_STEPS 150 -e DELPHI_SFT_STEPS 250 \
  -e DELPHI_NUM_GENERATIONS 16 -e DELPHI_BATCH_SIZE 8 -e DELPHI_LR 1e-5 \
  -- python launch_agentic.py
```

Swap `DELPHI_AGENT_MODE` (`t0`/`t1`/`t2`) and `DELPHI_SFT_STEPS` (`150` or `250`)
for the other stages. T0 and T1 are robust at `DELPHI_SFT_STEPS=150`; T2 wants
`250` for a sustained 1.0.

A CPU run is for **compile/import/unit checks only** — never validate a training
claim on CPU:

```bash
uv sync --frozen --no-group dev
JAX_PLATFORMS=cpu uv run python -c \
  "import agentic_common, agentic_sft, agentic_tools, train_agentic, launch_agentic"
```

---

## Where the findings live

- **[`REPORT.md`](REPORT.md) §9** — the agentic tool-use experience report: the
  T0/T1/T2 results table, *why* RL can't bootstrap the copy, and the four
  hard-won findings (SFT warm-up, gradient clipping, SFT over-collapse, train/RL
  prompt mismatch). §1–§8 cover the original tunix-on-iris feasibility verdict and
  the arithmetic/algebra results.
- **[`DESIGN.md`](DESIGN.md)** — the up-front design + A/B/C strategy trade study
  (port grug to nnx vs load Delphi-as-Qwen3 vs keep equinox) and the rollout plan.
- **[`AGENTS.md`](AGENTS.md)** — how to work in this directory: the file map, the
  recipe, the **invariants & gotchas that must not be regressed**, and how to add
  a new tool or curriculum stage.

Tracked as weaver issue #229 (feasibility) and issue #5 (agentic follow-on).
