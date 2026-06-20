# Working in `tunix-delphi-rl`

A guide for a future AI agent / contributor. This directory takes a **raw base
LM** (Delphi, a 447M Qwen3 with **no chat template**, that **never emits EOS** and
**can't emit JSON**) and bootstraps it — on
[google/tunix](https://github.com/google/tunix) + TPUs via marin's
[iris](https://github.com/marin-community/marin/tree/main/lib/iris) — through four
experiment families: non-agentic arithmetic GRPO, agentic calculator **tool use**,
single-turn **coding**, and multi-turn / curriculum **coding**. The crown-jewel
finding is a single recipe (SFT warm-up → RL on the same in-memory actor) and
*when each half matters*.

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
   lesson; §11 multi-turn. **Read this for the *why*; don't edit it.**
3. **This file** (`AGENTS.md`) — the directory map, the invariants that must not
   regress, and how to add a stage/tool.
4. **Pick an experiment** from the index below, read its `launch_*.py` entrypoint
   (at the repo root), then reproduce via [`training/pipeline.py`](training/pipeline.py)
   (the idempotent runner) or submit the matching `launch_*.py` on a TPU.

`uv run python ...` / `uv run pytest ...` everywhere — bare `python` is not on PATH.

---

## Directory layout

The repo is a small set of importable packages plus the iris launch entrypoints at
the root. Each package groups by **responsibility**; group new code by what it *is*,
not by which experiment uses it.

```
tunix-delphi-rl/
├── launch_*.py        iris ENTRYPOINTS — stay at the repo root (see invariant below)
├── models/            Delphi-as-Qwen3 loading + the RoPE patch
├── problems/          task/problem definitions (the "what to solve")
├── environments/      execution envs, the interpreter, tools, curriculum scheduler
├── training/          training drivers + shared training helpers + the runner
├── examples/          the "emit more cats" GRPO smoke task + its launcher
├── tests/             ALL tests (pytest); fast CPU units by default, slow ones gated
├── pyproject.toml     the validated tunix-native manifest + pytest config
└── *.md               README / REPORT / DESIGN / etc. (the narrative; read, don't churn)
```

**`models/`** — the model layer.
| module | what it does |
|---|---|
| `models/delphi_qwen3.py` | `delphi_config()` + `load_delphi()` (safetensors load, **hard 124/124 key-coverage assertion**, applies the RoPE patch) + `load_tokenizer()` (Llama-3, pad=eos). Exports `DELPHI_BOS_ID` / `DELPHI_EOS_ID`. |
| `models/delphi_patch.py` | `patch_tunix_rope_for_delphi()` — the worker-shippable RoPE monkeypatch (bakes in `rope_theta=500000` + Llama-3 `rope_scaling`) for exact HF parity on stock `google-tunix 0.1.7`. Called by `load_delphi`. |

**`problems/`** — task / problem definitions (no RL, no JAX accelerator needed).
| module | what it does |
|---|---|
| `problems/arithmetic.py` | The **non-agentic** arithmetic/algebra env (`build_arithmetic_dataset`, `answer_reward` / `format_reward` / `proximity_reward`, `metric_fn`); used by `train_delphi` and `train_agentic_port`. |
| `problems/coding_tasks.py` | The **held-out** fixed task ladder (tiers 0–5) — `(prompt, reference solution, gold stdout)`; the eval set. Tiers 0–4 (the §9 ladder); tier 5 the harder multi-turn (#8) eval set. |
| `problems/coding_problems.py` | The **curriculum** problem families (issue #8): per-level parameterized families with public/hidden test cases, `grade_problem`, `problem_reward`, `sample_problem` / `load_eval_problems`, prompt + feedback formatting. |

**`environments/`** — execution environments, the verifier, tools, and the scheduler.
| module | what it does |
|---|---|
| `environments/micropython.py` | A purely-functional, sandboxed, bounded tree-walking interpreter for a Python subset (`run(src) -> ExecResult`, never raises) — the execution env + verifier for coding. |
| `environments/coding_env.py` | Single-turn coding RL env: parameterized **task families** (anti-hardcode), `CODE_FEWSHOT` (shared SFT+RL prefix), the `END`-sentinel parser, the dense reward + solve/ran_ok/has_code metric, `code_segments`, `build_code_dataset`, `families_for_tiers`, greedy `evaluate_tasks`. |
| `environments/coding_agent_env.py` | **Multi-turn** coding env on tunix's *agentic* stack: `CODE_AGENT_SYSTEM_PROMPT`, `RunCodeAgent` + `CodeRunEnvironment` (the interpreter is the "tool", invoked per round), `code_agent_segments(rng, tiers, fix_prob)`, `build_code_agent_dataset`, `program_terminal_eos_tokens` (per-turn **END** stop), `evaluate_tasks_multiturn` (first vs best), `PassKResult`, best-across-rounds reward. |
| `environments/curriculum_env.py` | The **curriculum** coding env (issue #8): `TestCaseEnvironment` (test-case-graded, best-across-rounds), `build_curriculum_dataset` (level ramp driven by the scheduler), `solve_metric_fn`, `solve_segments`, `evaluate_problems_passk`, `load_eval_suite`, `CODE_SOLVE_SYSTEM_PROMPT`. |
| `environments/curriculum.py` | The fixed-cadence **curriculum scheduler** (issue #8): tiny deterministic state (highest unlocked level + per-level EMA), mastery gate, frontier-biased sampling, graduation. Pure Python (numpy + random). Used by `curriculum_env` + `train_curriculum`. |
| `environments/agentic_tools.py` | Tool-use shared components: `build_chain_dataset` (+ `build_t0/t1/t2`); the `T0/T1/T2_SYSTEM_PROMPT` few-shot demos; `CalcTextToolParser` (closing `)` optional); `DelphiToolAgent`; `CalcToolEnvironment` (copy-aware shaped reward); `t0_metric_fn`; `newline_terminal_eos_tokens`; `install_per_call_rollout_seed` (per-generation seed so the GRPO group isn't byte-identical). |

**`training/`** — training drivers and shared training helpers.
| module | what it does |
|---|---|
| `training/agentic_common.py` | Shared glue: `DelphiRawTextChatParser` (renders chat → **raw text**; `tool` role → `"Tool result: …"`; `generation_suffix="\n"`), and `clipped_adamw(lr)` (the global-norm-clipped AdamW used by **both** SFT and RL — invariant **B**). |
| `training/agentic_sft.py` | The SFT warm-up (stages 1+2). `chain_segments(rng, depth)`; `t0/t1/t2_segments`; `_encode_segments` → `(input_tokens, loss_mask, pad_mask)`; `build_sft_dataset` prepends a **masked `prompt_prefix`** (invariant **D**); `run_sft_warmup(model, ..., segment_fn=, prompt_prefix=)` runs `PeftTrainer` **in place** on the same nnx actor and returns it — used by all RL drivers. |
| `training/train_delphi.py` | The non-agentic GRPO harness (`DELPHI_STAGE / DELPHI_STEPS / DELPHI_LR`). The proven single-turn wiring that `train_coding` builds on. |
| `training/train_agentic.py` | Tool-use drivers: `train_agentic_port` (no-tool plumbing smoke), `_train_agentic_calc` (SFT warm-up → RLCluster/GRPOLearner → train), `train_agentic_t0/t1/t2`. Also `_NormalizingGRPOLearner` + metrics capture. |
| `training/train_coding.py` | Single-turn SFT warm-up → **Dr.GRPO** driver (`CODING_TIERS / CODING_SFT_STEPS / CODING_STEPS`). |
| `training/train_multiturn.py` | Multi-turn SFT warm-up → **Dr.GRPO** (agentic `GRPOConfig` with `advantage_estimator="drgrpo"` + `loss_agg_mode="sequence-mean-token-scale"`). Exports `_NormalizingGRPOLearner`, `_build_mesh` (reused by `train_curriculum`). |
| `training/train_curriculum.py` | Curriculum SFT warm-up → Dr.GRPO with the level-ramp scheduler (issue #8); per-level pass@k eval before/after RL. |
| `training/pipeline.py` | The idempotent reproduction runner: stages-as-data → `train_coding`/`train_multiturn` → per-stage results JSON (skips done stages). CLI `--experiment {coding,multiturn} --stage NAME|--all --results-dir --dry-run`. Run as `python -m training.pipeline`. |

**`examples/`** — `examples/toy_cats.py` + `examples/launch.py`: the "emit more cats"
GRPO smoke task and its iris launcher (plumbing, not an experiment family).

**`tests/`** — every test lives here (pytest). See **Tests** below.

---

## The iris launch entrypoints (a HARD invariant)

The `launch_*.py` files **stay at the repo root** and are submitted to iris as the
literal command `python launch_<name>.py`. iris snapshots the repo, `uv sync`s the
committed lock, and runs that exact command from the repo root; the repo root is on
`sys.path`, so a launcher's `from training.train_curriculum import …` resolves. **Do
not move the launchers into a package** and do not turn them into `-m` modules — that
breaks the submit command. The current launchers:

| launcher | experiment | example iris submit (the coordinator runs this; do NOT submit yourself) |
|---|---|---|
| `launch_delphi.py` | arithmetic GRPO | `… -e DELPHI_STAGE 0 -e DELPHI_STEPS 400 -e DELPHI_LR 1e-5 -- python launch_delphi.py` |
| `launch_agentic.py` | tool use | `… -e DELPHI_AGENT_MODE t2 -e DELPHI_SFT_STEPS 250 -e DELPHI_STEPS 150 -- python launch_agentic.py` |
| `launch_coding.py` | single-turn coding | `… -e CODING_SFT_STEPS 1000 -e CODING_STEPS 0 -- python launch_coding.py` |
| `launch_multiturn.py` | multi-turn coding | `… -e MT_TIERS 3,4,5 -e MT_SFT_STEPS 600 -e MT_STEPS 0 -- python launch_multiturn.py` |
| `launch_curriculum.py` | curriculum coding (#8) | `… -e CURRIC_STEPS 200 -e CURRIC_SFT_STEPS 200 -- python launch_curriculum.py` |

Full TPU flags (`--tpu v6e-4 --enable-extra-resources --extra tpu --region … --cpu 8
--memory 64GB --disk 60GB --max-retries N`) are in each launcher's docstring.

---

## Experiment index

| family | entrypoint(s) | what it does | how to run | REPORT.md |
|---|---|---|---|---|
| **Arithmetic GRPO** (non-agentic) | `training/train_delphi.py` / `launch_delphi.py` | Delphi computes the answer itself (no tool); exact/shaped reward over an answer-density curriculum. The feasibility study + the answer-space-density law. | iris: `launch_delphi.py` | §5, §7 |
| **Agentic TOOL USE** (#5) | `training/train_agentic.py` / `launch_agentic.py` (+ `training/agentic_common`, `training/agentic_sft`, `environments/agentic_tools`) | T0/T1/T2 chained `CALC(a * b)` calls to ~100% solve. **RL is essential** (amplifies the rare result-COPY). | iris: `launch_agentic.py` | §8 |
| **CODING single-turn** (#7) | `training/train_coding.py` / `launch_coding.py` (+ `environments/micropython`, `problems/coding_tasks`, `environments/coding_env`) | Delphi writes a Python program; `micropython` executes + grades it; SFT→Dr.GRPO; eval on a fixed 50-task ladder. **SFT does the work; Dr.GRPO marginal.** | `pipeline --experiment coding` **or** iris: `launch_coding.py` | §9 |
| **CODING multi-turn** (#8) | `training/train_multiturn.py` / `launch_multiturn.py` (+ `environments/coding_agent_env`) | write → run → read-output → revise; env returns the **best-across-rounds** grade; metric is **first-attempt vs best-across-rounds** solve. | `pipeline --experiment multiturn` **or** iris: `launch_multiturn.py` | §11 |
| **CODING curriculum** (#8) | `training/train_curriculum.py` / `launch_curriculum.py` (+ `environments/curriculum`, `environments/curriculum_env`, `problems/coding_problems`) | test-case-graded, curriculum-scheduled SFT→Dr.GRPO; per-level pass@1/pass@k on held-out instances before/after RL. | iris: `launch_curriculum.py` | §11 |

Smoke / plumbing (not an experiment family): `examples/toy_cats.py` +
`examples/launch.py` (the "emit more cats" GRPO smoke task).

---

## Tests

All tests are pytest under `tests/`. The **default run is fast + CPU-only**; tests
that download/load Delphi weights or run a JAX training loop are marked `slow` and
**deselected by default** (configured in `pyproject.toml`).

```bash
uv run pytest -q                          # the fast CPU unit suite (slow tests skipped)
uv run pytest tests/test_curriculum.py    # one file
JAX_PLATFORMS=cpu uv run pytest -m slow    # the slow gates (need model weights / accelerator)
```

| test file | covers | speed |
|---|---|---|
| `tests/test_micropython.py` | the interpreter language subset, error/safety surface, determinism | fast |
| `tests/test_coding_tasks.py` | every reference solution runs and reproduces its gold stdout | fast |
| `tests/test_coding_problems.py` | curriculum families: references solve their tests, partial credit, eval determinism, no test leakage | fast |
| `tests/test_coding_env.py` | single-turn families produce valid programs; parser/reward/metric; oracle solves the ladder | fast |
| `tests/test_coding_agent_env.py` | multi-turn parser, best-across-rounds env grade, first-vs-best metric, SFT segments, pass@k estimator | fast |
| `tests/test_curriculum_env.py` | curriculum env grade, metric, SFT segments, dataset level-ramp | fast |
| `tests/test_curriculum.py` | the scheduler math: advance/hold/force, frontier sampling, graduation | fast |
| `tests/test_smoke_cats.py` | the toy GRPO loop **learns** + the KV-cache is wired (`@slow`) | slow (JAX) |
| `tests/test_delphi_load.py` | Delphi load key-coverage + HF parity gate (`@slow`) | slow (weights) |

The fast units used to live in `if __name__ == "__main__":` self-check blocks inside
the source modules. **Do not add new `__main__` test blocks** — write a `tests/test_*.py`
instead. (The launchers and `training/pipeline.py` keep a real `if __name__ ==
"__main__": main()` — those are entrypoints, not tests.)

---

## The 3-stage recipe (what this directory implements)

Turning a raw base LM into a tool user / coder happens in three conceptual stages:

1. **SFT for token format** — teach the raw transcript token format
   (`Q: ... / CALC(...) / Tool result: ... / answer`, or `Task: ... <program> END
   / Tool result: ...`), with **per-turn loss masking**: train (mask 1) on the
   model's own turns and *not* (mask 0) on the env's lines — the env emits those at
   RL time and the model must learn to **copy/read** them, not produce them.
2. **SFT for tool/JSON calling** — teach the tool-call *surface* and the
   result-*copy*. Qwen JSON was OOD for this base LM (`tool_call_rate ≈ 0`), so the
   surface is a bare-text **`CALC(a * b)`** (tool use) or a program ended by an
   **`END`** sentinel (coding).
3. **RL curriculum** — staged group-relative RL: GRPO for tool use (T0 `a*b` → T1
   `a*b*c` → T2 `a*b*c*d`), Dr.GRPO for coding.

**In code, stages 1 & 2 are merged** into one SFT warm-up (`run_sft_warmup`,
`training/agentic_sft.py`): a single synthetic transcript per stage carries both the
line format and the surface, with per-turn masking. Split them only if the format and
the surface must come from different corpora. Stage 3 is `train_agentic_t0/t1/t2`,
`train_coding`, `train_multiturn`, or `train_curriculum`.

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
Guard: `run_sft_warmup` (`training/agentic_sft.py`), invoked when `sft_steps > 0`.
Never remove the warm-up from the tool stages.

### B. Gradient clipping is load-bearing — its absence is a *crash*

Unclipped multi-turn GRPO hits `inf`/`NaN` grads that surface as a libtpu
**`SIGSEGV`** (lr 2e-5 died ~step 3; lr 1e-5 ~step 99) — a crash that loses all
progress, not mere instability. `optax.chain(clip_by_global_norm(1.0), adamw(...))`
both eliminates the crash and stabilizes training. Guard:
`training.agentic_common.clipped_adamw` — the **single** source for the optimizer,
used by the SFT phase *and* every RL phase. Clip both; never swap in a bare `adamw`.

### C. SFT over-collapses if too strong — *but only for narrow targets*

For the **narrow** CALC copy, too much warm-up sharpens the policy onto a
degenerate continuation and drives `tool_call_rate → 0`; the sweet spot is
**~150–250 transcripts** (T0/T1 at 150; T2 at 250); ~400 collapses T0. For the
**broad** coding target, SFT instead scales **monotonically and plateaus** (45→48
across SFT 150→1500, no collapse) — more SFT buys coverage. *The right SFT amount
depends on how broad the target behavior is.* Guard: keep `DELPHI_SFT_STEPS` in the
150–250 band for tool stages; for coding, more is fine up to the plateau.

### D. Train/RL prompt mismatch silently breaks emission

SFT must use the **same prompt distribution** as the RL rollout. A naive warm-up
trained on `BOS Q:… → CALC` but RL prompts with `‹few-shot› Q:…` — a context the
warm-up never saw — so the model dropped the `CALC(`. Benign for single-call T0;
**corrupts chained T1/T2 turn-1**. Fix: **prepend the masked few-shot prompt to
every SFT transcript** (`prompt_prefix` in `run_sft_warmup` / `build_sft_dataset`):
`T{1,2}_SYSTEM_PROMPT` for tool use, `CODE_FEWSHOT` for single-turn coding,
`CODE_AGENT_SYSTEM_PROMPT` for multi-turn. Generalizable lesson: warm-up transcripts
must match the RL prompt distribution, **prefix included**.

### Dr.GRPO drop-in (the coding RL algorithm)

Both coding experiments use **Dr.GRPO** instead of GRPO (more robust on a tiny
actor): advantage = group-mean-centered with **no std division**, loss
**constant-normalized**. Single-turn: `DrGRPOLearner`/`DrGRPOConfig`
(`training/train_coding.py`). Multi-turn / curriculum: the *agentic* `GRPOConfig`
with `advantage_estimator="drgrpo"` + `loss_agg_mode="sequence-mean-token-scale"`
(`training/train_multiturn.py`, `training/train_curriculum.py`) — same math on the
agentic learner. It is a drop-in for the GRPO learner/config.

### Base-LM surface constraints

- **No chat template.** Use `DelphiRawTextChatParser` (raw text), never the stock
  Qwen template (it emits out-of-vocab `<|im_start|>`).
- **Never emits EOS.** Single-line turns (CALC) stop on a **digit/`)`-terminated
  newline** (`newline_terminal_eos_tokens`) plus bare `"\n"`. **Multi-line turns
  (programs) cannot stop on a newline** — they stop on the **`END` sentinel token**
  (`program_terminal_eos_tokens`); single-turn coding generates to the budget then
  `extract_program` cuts at `END`. The chat parser's `generation_suffix="\n"` ensures
  content before the first newline.
- **Can't reliably emit Qwen JSON.** The tool surface is bare-text `CALC(a * b)`.
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
constant. Both stacks **randomize per prompt** — CALC random operands, coding's
**task families** with random params — so the gold varies and the model must write a
*general* program / emit the *right* operands. Eval prompts strip answer-leaking
hints (`strip_answer_hint`). Keep new tasks parameterized.

### CPU is for compile/import/unit checks ONLY

CPU validates `import` / compile / pure-logic units (parsers, segment encoding,
dataset shapes, the `tests/` fast suite, `python -m training.pipeline --dry-run`).
**Never** validate a training objective or a learning claim on CPU — all training
objectives are validated on **TPU (v6e, free on iris)**. Use TPU aggressively
(parallel sweeps); use CPU only to catch breakage before paying for a TPU.

---

## The cross-experiment lesson: when is RL essential vs when does SFT suffice?

The follow-ons are the **same recipe** pointed at two behaviors, and they come out
*opposite* (`REPORT.md` §10):

| | §8 tool use (copy the result) | §9 coding (write the program) |
|---|---|---|
| target behavior | **narrow** (one copy-forward step) | **broad** (many program families) |
| base-model sample rate | rare → SFT can't fully cover it | demonstrable → SFT covers it |
| SFT scaling | over-collapses past ~250 (invariant C) | monotone, plateaus, no collapse |
| RL role | **essential** — amplifies the rare copy | **marginal** — no headroom left |

**Deciding rule.** RL earns its keep only for a *narrow* behavior that must be
amplified from rare base-model samples. When the target is *fully demonstrable by
SFT*, SFT alone wins and RL is moot. Both hit the same wall ("RL only sharpens what
the base policy already puts mass on") and resolve it the same way (an SFT warm-up).

**Multi-turn / curriculum coding (#8) is the test of that rule.** Its hypothesis is
that on a *harder* task set the first-attempt solve rate is low even after SFT, and
the win comes from **iterating on execution feedback** — read the interpreter's
stdout/error and write a corrected program. That repair behavior is only weakly
demonstrable by SFT (`code_agent_segments` makes it in-distribution but **rare** via
`fix_prob`), so RL should amplify it — flipping §9's result back to "RL essential".
See `REPORT.md` §11.

---

## How to add a new curriculum stage (tool use)

Stages differ **only in chain depth** (number of `*` operands):

1. **Operand names** — `environments.agentic_tools._OPERAND_NAMES` has `a,b,c,d`
   (depth ≤ 3); add `"e"` for depth 4.
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

1. **Tool** — reuse a stock `tunix.rl.agentic.tools` tool or write one; register it.
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

Add a `Family` in `environments/coding_env.py` (single-turn) or a level family in
`problems/coding_problems.py` (curriculum) — parameterized, gold computed, never
hardcoded — and, if it should be evaluated, a fixed `Task` in
`problems/coding_tasks.py`. Run the relevant fast test
(`uv run pytest tests/test_coding_env.py` / `tests/test_coding_problems.py` /
`tests/test_coding_tasks.py`) — they assert every family produces a valid bounded
program and the oracle solves the ladder. Coverage (covering the eval's task *types*),
not optimization, is the lever (§9.4).

---

## Style & verification

- **2-space indent**, Google-style docstrings. Keep the explanatory docstrings that
  record *why* (the findings, the BPE / EOS / seed gotchas) — condense if verbose,
  but don't strip the rationale.
- Refactors must be **behavior-preserving**. Preserve the public entrypoints
  (`train_agentic_port` / `train_agentic_t0/t1/t2`, `train_coding`, `train_multiturn`,
  `train_curriculum`) and the env-var interfaces; use **absolute package imports**
  (e.g. `from environments.coding_env import …`); update **all** call sites if you
  rename internal helpers (`grep` the dir).
- **Verify before reporting** (all CPU, no TPU):
  ```bash
  JAX_PLATFORMS=cpu uv run python -c "import environments.coding_agent_env, training.train_multiturn, launch_multiturn, training.train_coding, environments.coding_env, problems.coding_tasks"
  JAX_PLATFORMS=cpu uv run python -c "import training.agentic_common, training.agentic_sft, environments.agentic_tools, training.train_agentic, launch_agentic"
  JAX_PLATFORMS=cpu uv run python -c "import training.train_curriculum, launch_curriculum, environments.curriculum, environments.curriculum_env"
  JAX_PLATFORMS=cpu uv run pytest -q                                   # fast CPU unit suite
  JAX_PLATFORMS=cpu uv run python -m training.pipeline --dry-run --experiment multiturn
  ```
  Do **not** run TPU jobs or `iris` from here — the coordinator submits those.
