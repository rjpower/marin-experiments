# Working in `tunix-delphi-rl`

A guide for a future AI agent / contributor. This directory takes a **raw base
LM** (Delphi, a 447M Qwen3 with **no chat template**, that **never emits EOS** and
**can't emit JSON**) and bootstraps it — on
[google/tunix](https://github.com/google/tunix) + TPUs via marin's
[iris](https://github.com/marin-community/marin/tree/main/lib/iris) — through four
experiment families: non-agentic arithmetic GRPO, agentic calculator **tool use**,
single-turn **coding**, and multi-turn **coding**. The crown-jewel finding is a
single recipe (SFT warm-up → RL on the same in-memory actor) and *when each half
matters*.

---

## START HERE

**Orientation (one paragraph).** Everything in this repo is the same loop pointed
at harder behaviors: load Delphi as a tunix-native Qwen3 (exact HF parity via a
RoPE monkeypatch), give it a few-shot prompt + an SFT warm-up to put a behavior
*in distribution*, then run group-relative RL (GRPO / Dr.GRPO) on a TPU to amplify
it. The binding constraint was **never** the RL math or the iris/TPU plumbing — it
was always the base LM's format/copy priors, and a **distribution-matched SFT
warm-up** is the lever. The most useful result is *cross-experiment*: RL is
essential only to amplify a **narrow** behavior the base LM rarely samples (the
tool-result copy, §8); when the target is **fully demonstrable by SFT** (writing a
program, §9), SFT does the work and RL is marginal.

**Reading order (do these in sequence):**

1. [`README.md`](README.md) — the headline result + how to run the agentic tool use.
2. [`REPORT.md`](REPORT.md) — the experience report (the experiment arc + lessons).
   §1–§7 feasibility + arithmetic; §8 tool use; §9 coding; §10 the cross-experiment
   lesson; §11 multi-turn (in progress). **Read this for the *why*; don't edit it.**
3. **This file** (`AGENTS.md`) — the file map, the invariants that must not regress,
   and how to add a stage/tool.
4. **Pick an experiment** from the index below, read its entrypoint script, then
   reproduce via [`pipeline.py`](pipeline.py) (the idempotent runner) or submit the
   matching `launch_*.py` on a TPU.

**Reproduce, don't re-derive.** [`pipeline.py`](pipeline.py) codifies the *rounds
of tuning* as data and runs them idempotently (skips any stage whose results JSON
exists). Inspect the plan on CPU:

```bash
uv run python pipeline.py --dry-run --experiment coding      # the §9 SFT ladder + one SFT->RL round
uv run python pipeline.py --dry-run --experiment multiturn   # few-shot -> SFT -> SFT+Dr.GRPO (§11)
```

Then on a TPU host (Delphi at `$DELPHI_MODEL_DIR`): `uv run python pipeline.py
--experiment coding --stage sft1000` or `--all`. **Do not run training from a
CPU/your laptop** and **do not submit iris jobs yourself** — the coordinator does
that; you reproduce by selecting a stage.

`uv run python ...` everywhere — bare `python` is not on PATH.

---

## Experiment index

| family | entrypoint(s) | what it does | how to run | REPORT.md |
|---|---|---|---|---|
| **Arithmetic / algebra GRPO** (non-agentic) | `train_delphi.py` / `launch_delphi.py` | Delphi computes the answer itself (no tool); exact/shaped reward over an answer-density curriculum (add → algebra). The feasibility study + the answer-space-density law. | iris: `-e DELPHI_STAGE {0..3} -e DELPHI_STEPS 400 -e DELPHI_LR 1e-5 -- python launch_delphi.py` | §5, §7 |
| **Agentic TOOL USE** (#5) | `train_agentic.py` / `launch_agentic.py` (+ `agentic_common`, `agentic_sft`, `agentic_tools`) | T0/T1/T2 chained `CALC(a * b)` calls — each tool output feeds the next call's argument — to ~100% solve. **RL is essential** (amplifies the rare result-COPY). | iris: `-e DELPHI_AGENT_MODE {port,t0,t1,t2} -e DELPHI_SFT_STEPS {150,250} -e DELPHI_STEPS 150 -- python launch_agentic.py` | §8 |
| **Agentic CODING single-turn** (#7) | `train_coding.py` / `launch_coding.py` (+ `micropython`, `coding_tasks`, `coding_env`) | Delphi writes a Python program; `micropython` executes + grades it; SFT→Dr.GRPO on 41 parameterized families; eval on a fixed 50-task ladder. **SFT does the work; Dr.GRPO marginal.** | `pipeline.py --experiment coding` **or** iris: `-e CODING_SFT_STEPS 1000 -e CODING_STEPS 0 -- python launch_coding.py` | §9 |
| **Agentic CODING multi-turn** (#8) | `train_multiturn.py` / `launch_multiturn.py` (+ `coding_agent_env`) | write → run → read-output → revise, up to `rounds`; env returns the **best-across-rounds** grade; metric is **first-attempt vs best-across-rounds** solve. The regime where RL *should* finally beat SFT (in progress). | `pipeline.py --experiment multiturn` **or** iris: `-e MT_TIERS 3,4,5 -e MT_SFT_STEPS 600 -e MT_STEPS 0 -- python launch_multiturn.py` | §11 |

Smoke / plumbing (not an experiment family): `toy_cats.py` + `launch.py` (the
"emit more cats" GRPO smoke task; `test_smoke_cats.py`). `test_delphi_load.py`
(HF-parity load gate). `test_micropython.py` (90 interpreter unit tests).

---

## The 3-stage recipe (what this directory implements)

Turning a raw base LM into a tool user / coder happens in three conceptual stages:

1. **SFT for token format** — teach the raw transcript token format
   (`Q: ... / CALC(...) / Tool result: ... / answer`, or `Task: ... <program> END
   / Tool result: ...`), with **per-turn loss masking**: train (mask 1) on the
   model's own turns (its `CALC(...)`/program + the answer), and *not* (mask 0) on
   the env's lines (`Q:`, `Task:`, `Tool result:`) — the env emits those at RL
   time and the model must learn to **copy/read** them, not produce them.
2. **SFT for tool/JSON calling** — teach the tool-call *surface* and the
   result-*copy*. Qwen JSON was OOD for this base LM (`tool_call_rate ≈ 0`), so the
   surface is a bare-text **`CALC(a * b)`** (tool use) or a program ended by an
   **`END`** sentinel (coding).
3. **RL curriculum** — staged group-relative RL: GRPO for tool use (T0 `a*b` → T1
   `a*b*c` → T2 `a*b*c*d`), Dr.GRPO for coding.

**In code, stages 1 & 2 are merged** into one SFT warm-up (`run_sft_warmup`,
`agentic_sft.py`): a single synthetic transcript per stage carries both the line
format and the surface, with per-turn masking. Split them only if the format and
the surface must come from different corpora. Stage 3 is `train_agentic_t0/t1/t2`,
`train_coding`, or `train_multiturn`.

---

## File map

| file | what it does |
|---|---|
| `agentic_common.py` | Shared glue: `DelphiRawTextChatParser` (renders chat messages to **raw text** — Delphi has no chat template; `tool` role → `"Tool result: …"`; `generation_suffix="\n"` so a turn starts on a fresh line), and `clipped_adamw(lr)` (the global-norm-clipped AdamW used by **both** SFT and RL — invariant **B**). |
| `agentic_sft.py` | The SFT warm-up (stages 1+2). `chain_segments(rng, depth)` builds one masked transcript; `t0/t1/t2_segments` are named wrappers. `_encode_segments` tokenizes to `(input_tokens, loss_mask, pad_mask)`. `build_sft_dataset` prepends a **masked `prompt_prefix`** (invariant **D**). `run_sft_warmup(model, ..., segment_fn=, prompt_prefix=)` runs `PeftTrainer` **in place** on the same nnx actor and returns it — used by all three RL drivers. |
| `agentic_tools.py` | Tool-use shared components: `build_chain_dataset` (+ `build_t0/t1/t2` wrappers); the `T0/T1/T2_SYSTEM_PROMPT` few-shot demos; `CalcTextToolParser` (closing `)` optional); `DelphiToolAgent`; `CalcToolEnvironment` (copy-aware shaped reward); `t0_metric_fn`; `newline_terminal_eos_tokens` (digit/`)`-terminated-newline stop); `install_per_call_rollout_seed` (per-generation seed so the GRPO group isn't byte-identical). |
| `train_agentic.py` | Tool-use drivers: `train_agentic_port` (no-tool plumbing smoke), `_train_agentic_calc` (SFT warm-up → RLCluster/GRPOLearner → train), `train_agentic_t0/t1/t2` (per-stage wrappers). Also `_NormalizingGRPOLearner` + metrics capture. |
| `launch_agentic.py` | iris entrypoint, `DELPHI_AGENT_MODE={port,t0,t1,t2}`. |
| `delphi_qwen3.py` | `delphi_config()` + `load_delphi()` (safetensors load, **hard 124/124 key-coverage assertion**, applies the RoPE patch) + `load_tokenizer()` (Llama-3, pad=eos). Exports `DELPHI_BOS_ID` / `DELPHI_EOS_ID`. |
| `delphi_patch.py` | `patch_tunix_rope_for_delphi()` — the worker-shippable RoPE monkeypatch (bakes in `rope_theta=500000` + Llama-3 `rope_scaling`) for exact HF parity on stock `google-tunix 0.1.7`. Called by `load_delphi`. |
| `arithmetic.py` | The **non-agentic** arithmetic/algebra env (`build_arithmetic_dataset`, `answer_reward` / `format_reward` / `proximity_reward`, `metric_fn`); used by `train_delphi` and `train_agentic_port`. |
| `train_delphi.py` / `launch_delphi.py` | The non-agentic GRPO harness + iris entrypoint (`DELPHI_STAGE / DELPHI_STEPS / DELPHI_LR / ...`). Also the proven single-turn wiring that `train_coding` builds on. |
| `micropython.py` | A purely-functional, sandboxed, bounded tree-walking interpreter for a Python subset (`run(src) -> ExecResult`, never raises) — the execution env + verifier for coding. 90 tests in `test_micropython.py`. |
| `coding_tasks.py` | The **held-out** fixed task ladder (tiers 0–5) — `(prompt, reference solution, gold stdout)`; the eval set. Tiers 0–4 (the §9 ladder); tier 5 is the harder multi-turn (#8) eval set. |
| `coding_env.py` | Single-turn coding RL env: parameterized **task families** (anti-hardcode, like CALC random operands), `CODE_FEWSHOT` (shared SFT+RL prefix), the `END`-sentinel parser, the dense reward + solve/ran_ok/has_code metric, `code_segments`, `build_code_dataset`, `families_for_tiers`, and greedy `evaluate_tasks` (hints stripped). Has a CPU `__main__` self-check. |
| `train_coding.py` / `launch_coding.py` | Single-turn SFT warm-up → **Dr.GRPO** driver + iris entrypoint (`CODING_TIERS / CODING_SFT_STEPS / CODING_STEPS / ...`). |
| `coding_agent_env.py` | **Multi-turn** coding env on tunix's *agentic* stack: `CODE_AGENT_SYSTEM_PROMPT` (two demos, the 2nd a read-output-and-FIX), `RunCodeAgent` + `CodeRunEnvironment` (the interpreter is the "tool", invoked per round), `code_agent_segments(rng, tiers, fix_prob)` (a MINORITY of transcripts show a fix), `build_code_agent_dataset`, `program_terminal_eos_tokens` (per-turn **END** stop), `evaluate_tasks_multiturn` (first vs best). `SOLVE_REWARD_THRESHOLD`, best-across-rounds reward. CPU `__main__` self-check. |
| `train_multiturn.py` / `launch_multiturn.py` | Multi-turn SFT warm-up → **Dr.GRPO** (Dr.GRPO knobs on the *agentic* `GRPOConfig`: `advantage_estimator="drgrpo"`, `loss_agg_mode="sequence-mean-token-scale"`) + iris entrypoint (`MT_TIERS / MT_EVAL_TIERS / MT_ROUNDS / MT_SFT_STEPS / MT_STEPS / ...`). |
| `pipeline.py` | The idempotent reproduction runner: stages-as-data → `train_coding`/`train_multiturn` → per-stage results JSON (skips done stages). CLI `--experiment {coding,multiturn} --stage NAME|--all --results-dir --dry-run`. |
| `toy_cats.py` / `launch.py` | The "emit more cats" toy GRPO smoke task + entrypoint. |
| `test_*.py` | Real gates: `test_micropython.py` (90 interpreter units), `test_delphi_load.py` (HF parity), `test_smoke_cats.py` (toy GRPO learns). |
| `pyproject.toml`, `uv.lock` | The validated tunix-native manifest (no `marin-levanter`; see `DESIGN.md` §4). |

---

## Invariants & gotchas — DO NOT regress these

Hard-won (see `REPORT.md` §8 for the full rationale). Each has a guard in the code;
keep it.

### A. RL learns the tool *call*, not the result *copy* — so SFT-warm-up first

Copying an injected `Tool result: X` (into the next call or the final answer) is
**out-of-distribution** for the base LM — sampled too rarely for GRPO to amplify
(plain T0 RL drove `arg_acc` → ~0.99 but `solve_ratio` peaked ~0.1 then
**collapsed**). The fix is a short **SFT warm-up** that makes call+copy
in-distribution *before* RL, on the **same in-memory nnx actor** (no checkpoint
round-trip — both phases mutate the same module and `RLCluster` re-shards it).
Guard: `run_sft_warmup` (`agentic_sft.py`), invoked when `sft_steps > 0`. Never
remove the warm-up from the tool stages.

### B. Gradient clipping is load-bearing — its absence is a *crash*

Unclipped multi-turn GRPO hits `inf`/`NaN` grads that surface as a libtpu
**`SIGSEGV`** (lr 2e-5 died ~step 3; lr 1e-5 ~step 99) — a crash that loses all
progress, not mere instability. `optax.chain(clip_by_global_norm(1.0), adamw(...))`
both eliminates the crash and stabilizes training. Guard:
`agentic_common.clipped_adamw` — the **single** source for the optimizer, used by
the SFT phase *and* every RL phase (tool use, single-turn coding, multi-turn).
Clip both; never swap in a bare `adamw`.

### C. SFT over-collapses if too strong — *but only for narrow targets*

For the **narrow** CALC copy, too much warm-up sharpens the policy onto a
degenerate continuation and drives `tool_call_rate → 0`; the sweet spot is
**~150–250 transcripts** (T0/T1 at 150; T2 at 250); ~400 collapses T0. For the
**broad** coding target (41 families), SFT instead scales **monotonically and
plateaus** (45→48 across SFT 150→1500, no collapse) — more SFT buys coverage. *The
right SFT amount depends on how broad the target behavior is.* Guard: keep
`DELPHI_SFT_STEPS` in the 150–250 band for tool stages; for coding, more is fine up
to the plateau.

### D. Train/RL prompt mismatch silently breaks emission

SFT must use the **same prompt distribution** as the RL rollout. A naive warm-up
trained on `BOS Q:… → CALC` but RL prompts with `‹few-shot› Q:…` — a context the
warm-up never saw — so the model dropped the `CALC(`. Benign for single-call T0;
**corrupts chained T1/T2 turn-1**. Fix: **prepend the masked few-shot prompt to
every SFT transcript** (`prompt_prefix` in `run_sft_warmup` /
`build_sft_dataset`): `T{1,2}_SYSTEM_PROMPT` for tool use, `CODE_FEWSHOT` for
single-turn coding, `CODE_AGENT_SYSTEM_PROMPT` for multi-turn. Generalizable
lesson: warm-up transcripts must match the RL prompt distribution, **prefix
included**.

### Dr.GRPO drop-in (the coding RL algorithm)

Both coding experiments use **Dr.GRPO** instead of GRPO (more robust on a tiny
actor): advantage = group-mean-centered with **no std division**, loss
**constant-normalized** (no per-response length bias). Single-turn:
`DrGRPOLearner`/`DrGRPOConfig` (`train_coding.py`). Multi-turn: the *agentic*
`GRPOConfig` with `advantage_estimator="drgrpo"` +
`loss_agg_mode="sequence-mean-token-scale"` (`train_multiturn.py`) — same math on
the agentic learner. It is a drop-in for the GRPO learner/config.

### Base-LM surface constraints

- **No chat template.** Use `DelphiRawTextChatParser` (raw text), never the stock
  Qwen template (it emits out-of-vocab `<|im_start|>`).
- **Never emits EOS.** Single-line turns (CALC) stop on a **digit/`)`-terminated
  newline** (`newline_terminal_eos_tokens`) plus bare `"\n"`. **Multi-line turns
  (programs) cannot stop on a newline** — they stop on the **`END` sentinel token**
  (`program_terminal_eos_tokens`); single-turn coding has no per-turn stop at all
  and generates to the budget, then `extract_program` cuts at `END`. The chat
  parser's `generation_suffix="\n"` ensures content before the first newline.
- **Can't reliably emit Qwen JSON.** The tool surface is bare-text `CALC(a * b)`
  (`tool_call_rate ≈ 0` for JSON on TPU).
- **Llama-3 BPE fuses `)\n`.** A correctly-closed `CALC(a * b)` is recorded as
  `CALC(a * b`. The parser, `arg_reward`, and metrics treat the closing `)` as
  **optional** (`_CALC_PREFIX_RE`); don't tighten them.
- **GRPO group must vary.** tunix 0.1.7 generates each group member with a fixed
  `RolloutConfig.seed`, so all `num_generations` samples would be byte-identical →
  zero advantage → no gradient. Guard: `install_per_call_rollout_seed`. Keep it.
- **`kv_cache_size ≥ max_prompt_length + max_tokens_to_generate`** or the sampler
  hard-errors (the drivers add `+ 8` headroom).
- **Actor storage must be fp32** (bf16 rounds small Adam updates to zero). Compute
  can be bf16 via `config.dtype`; the reference (if `beta > 0`) is bf16.

### Anti-hardcoding: parameterize the task

A fixed task + an exact-output reward lets the model "solve" by memorizing the
constant. Both stacks **randomize per prompt** — CALC random operands, coding's 41
**task families** with random params — so the gold varies and the model must write
a *general* program / emit the *right* operands. Eval prompts strip answer-leaking
hints (`strip_answer_hint`). Keep new tasks parameterized.

### CPU is for compile/import/unit checks ONLY

CPU validates `import` / compile / pure-logic units (parsers, segment encoding,
dataset shapes, `coding_env.py` / `coding_agent_env.py` `__main__` self-checks,
`pipeline.py --dry-run`). **Never** validate a training objective or a learning
claim on CPU — all training objectives are validated on **TPU (v6e, free on
iris)**. Use TPU aggressively (parallel sweeps); use CPU only to catch breakage
before paying for a TPU.

---

## The cross-experiment lesson: when is RL essential vs when does SFT suffice?

The two follow-ons are the **same recipe** pointed at two behaviors, and they come
out *opposite* (`REPORT.md` §10):

| | §8 tool use (copy the result) | §9 coding (write the program) |
|---|---|---|
| target behavior | **narrow** (one copy-forward step) | **broad** (41 program families) |
| base-model sample rate | rare → SFT can't fully cover it | demonstrable → SFT covers it |
| SFT scaling | over-collapses past ~250 (invariant C) | monotone, plateaus, no collapse |
| RL role | **essential** — amplifies the rare copy | **marginal** — no headroom left |

**Deciding rule.** RL earns its keep only for a *narrow* behavior that must be
amplified from rare base-model samples. When the target is *fully demonstrable by
SFT*, SFT alone wins and RL is moot. Both hit the same wall ("RL only sharpens what
the base policy already puts mass on") and resolve it the same way (an SFT warm-up);
they differ only in whether SFT can *finish* the job or merely *start* it.

**Multi-turn coding (#8) is the test of that rule.** Its hypothesis is that on a
*harder* task set the first-attempt solve rate is low even after SFT, and the win
comes from **iterating on execution feedback** — read the interpreter's stdout/error
and write a corrected program. That repair behavior is only weakly demonstrable by
SFT (`code_agent_segments` makes it in-distribution but **rare** via `fix_prob`), so
RL should amplify it — flipping §9's result back to "RL essential". Mechanics:

- **write → run → revise loop** — up to `rounds` rounds; the `micropython`
  interpreter is the "tool", invoked by the env each round (not called in-band — the
  program *is* the action); its stdout/error is injected as `Tool result:`.
- **best-across-rounds env reward** — the learner's `reward_fns` only see the first
  assistant turn, so the multi-round grade lives in the env: per-step rewards are 0
  and `CodeRunEnvironment._compute_final_reward` returns the dense
  **best-across-rounds** grade (learner built with `reward_fns=None`). RL is rewarded
  for *reaching* a correct program within the budget.
- **per-turn END stop** — programs are multi-line, so each turn stops on the `END`
  sentinel (`program_terminal_eos_tokens`); without it the first turn eats the whole
  episode budget.
- **first-vs-best-solve metric** — `evaluate_tasks_multiturn` reports first-attempt
  vs best-across-rounds solve per tier; the gap (and how RL grows it) is the headline.

This is **in progress; no results claimed.** `REPORT.md` §11.

---

## How to add a new curriculum stage (tool use)

Stages differ **only in chain depth** (number of `*` operands):

1. **Operand names** — `agentic_tools._OPERAND_NAMES` has `a,b,c,d` (depth ≤ 3); add
   `"e"` for depth 4.
2. **System prompt** — add a `T3_SYSTEM_PROMPT` few-shot demo (two worked depth-4
   transcripts; byte-exact line format).
3. **Dataset / segments** — already parameterized: `build_chain_dataset(...,
   depth=4)` and `chain_segments(rng, depth=4)` just work; add `build_t3_dataset` /
   `t3_segments` wrappers.
4. **Driver** — add `train_agentic_t3` mirroring `t2`: `dataset_builder=build_t3_dataset`,
   `sft_segment_fn=t3_segments`, `system_prompt=T3_SYSTEM_PROMPT`,
   `sft_prompt_prefix=T3_SYSTEM_PROMPT` (invariant **D**), `env_max_steps=5`, bumped
   `max_prompt_length` / `max_tokens_to_generate` / `sft_max_seq_len`.
5. **Launcher** — add `"t3"` to `_tool_train_fns` + the `DELPHI_AGENT_MODE` set.
6. **Budget** — start `DELPHI_SFT_STEPS=250` (deeper chains want slightly more).

## How to add a new tool

1. **Tool** — reuse a stock `tunix.rl.agentic.tools` tool or write one; register it
   in the tool map.
2. **Surface + parser** — a **bare-text** surface (not JSON), parser duck-typing
   `CalcTextToolParser`; keep parsing lenient about the fused trailing newline.
3. **Few-shot demos** — the demonstrations carry the format; add a system prompt.
4. **Agent** — if the result needs a specific rendering, mirror
   `DelphiToolAgent._observation_to_messages`.
5. **Reward** — give a **dense** copy/solve signal (sparse "final == gold" is too
   sparse to bootstrap — see `CalcToolEnvironment`).
6. **SFT** — add a `*_segments` builder and prepend the matching `prompt_prefix`
   (invariant **D**).

## How to add a coding task / family

Add a `Family` in `coding_env.py` (parameterized — random params per prompt, gold
computed, never hardcoded) and, if it should be evaluated, a fixed `Task` in
`coding_tasks.py`. Run the CPU self-check (`uv run python coding_env.py`) — it
asserts every family produces a valid bounded program and the oracle solves the
ladder. Coverage (covering the eval's task *types*), not optimization, is the lever
(§9.4): the SFT→50/50 came from broadening families, not more RL.

---

## Style & verification

- **2-space indent**, Google-style docstrings. Keep the explanatory docstrings that
  record *why* (the four findings, the BPE / EOS / seed gotchas) — condense if
  verbose, but don't strip the rationale.
- Refactors of the tool/SFT/coding code must be **behavior-preserving**. Preserve
  the public entrypoints (`train_agentic_port` / `train_agentic_t0/t1/t2`,
  `train_coding`, `train_multiturn`) and the env-var interfaces; update **all** call
  sites if you rename internal helpers (`grep` the dir).
- **Verify before reporting** (all CPU, no TPU):
  ```bash
  uv run python -c "import coding_agent_env, train_multiturn, launch_multiturn, train_coding, coding_env, coding_tasks"
  uv run python -c "import agentic_common, agentic_sft, agentic_tools, train_agentic, launch_agentic"
  uv run python coding_env.py            # self-check OK
  uv run python coding_agent_env.py      # self-check OK
  uv run python -m pytest test_micropython.py -q
  uv run python -c "import pipeline" && uv run python pipeline.py --dry-run --experiment multiturn
  ```
  Do **not** run TPU jobs or `iris` from here — the coordinator submits those.
