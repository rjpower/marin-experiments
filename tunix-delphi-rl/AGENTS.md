# Working in `tunix-delphi-rl`

A guide for a future AI agent / contributor. This directory bootstraps a **raw
base LM** (Delphi, a 447M Qwen3 with **no chat template**, that **never emits
EOS** and **can't emit JSON**) into an **agentic tool user** on
[google/tunix](https://github.com/google/tunix) + TPUs via marin's
[iris](https://github.com/marin-community/marin/tree/main/lib/iris).

Read [`README.md`](README.md) for the result and how to run it, and
[`REPORT.md`](REPORT.md) §9 for the full rationale behind the invariants below.
The non-agentic feasibility study (arithmetic/algebra GRPO) is in `REPORT.md`
§1–§8 and `DESIGN.md`.

---

## The 3-stage recipe (what this directory implements)

Turning a raw base LM into a tool user happens in three conceptual stages:

1. **SFT for token format** — teach the raw transcript token format
   (`Q: ... / CALC(...) / Tool result: ... / answer`), with **per-turn loss
   masking**: train (mask 1) on the model's own turns (`CALC(...)` + the final
   answer), and *not* (mask 0) on the env's lines (`Q:`, `Tool result:`) — the
   env emits those at RL time and the model must learn to **copy** them, not
   produce them.
2. **SFT for tool/JSON calling** — teach the tool-call *surface* and the
   result-*copy*. Qwen JSON was OOD for this base LM (`tool_call_rate ≈ 0`), so
   the surface is a bare-text **`CALC(a * b)`**.
3. **RL curriculum** — staged GRPO by chaining depth: **T0** `a*b` (1 call) →
   **T1** `a*b*c` (2 chained) → **T2** `a*b*c*d` (3 chained).

**In code, stages 1 & 2 are merged** into one SFT warm-up (`run_sft_warmup`,
`agentic_sft.py`): a single synthetic transcript per stage carries both the line
format and the `CALC`/copy surface, with per-turn masking. Split them only if the
format and the surface must come from different corpora (e.g. generic transcripts
+ a small tool-specific set, or a second tool with a different surface). Stage 3
is `train_agentic_t0/t1/t2`, each a thin wrapper over `_train_agentic_calc`.

---

## File map

| file | what it does |
|---|---|
| `agentic_common.py` | Shared glue: `DelphiRawTextChatParser` (renders chat messages to **raw text** — Delphi has no chat template; `tool` role → `"Tool result: …"`; `generation_suffix="\n"` for the tool stages so a turn starts on a fresh line), and `clipped_adamw(lr)` (the global-norm-clipped AdamW used by **both** SFT and RL — see invariant **B**). |
| `agentic_sft.py` | The SFT warm-up (stages 1+2). `chain_segments(rng, depth)` builds one masked transcript (depth 1/2/3 = T0/T1/T2); `t0/t1/t2_segments` are named wrappers. `_encode_segments` tokenizes a transcript into `(input_tokens, loss_mask, pad_mask)` (BOS-prefixed, right-padded with pad=eos). `_SFTSource` / `build_sft_dataset` prepend a **masked `prompt_prefix`** (invariant **D**). `run_sft_warmup(model, ...)` runs `PeftTrainer` **in place** on the same nnx actor and returns it. |
| `agentic_tools.py` | The tool stages' shared, **depth-parameterized** components: `build_chain_dataset(n, seed, bs, depth)` (+ `build_t0/t1/t2_dataset` wrappers) emitting `prompts` / operand / `answer` columns; the `T0/T1/T2_SYSTEM_PROMPT` few-shot demos; `CalcTextToolParser` (parses `CALC(a * b)`, closing `)` optional); `DelphiToolAgent` (suppressed tool docs + task-as-user-turn); `CalcToolEnvironment` (copy-aware shaped reward); `arg_reward` / `format_reward`; `t0_metric_fn` (`tool_call_rate` / `arg_acc` / `solve_ratio`); `newline_terminal_eos_tokens` (the digit/`)`-terminated-newline stop); `install_per_call_rollout_seed` (per-generation seed so the GRPO group isn't byte-identical). |
| `train_agentic.py` | The training drivers. `train_agentic_port` (single-turn no-tool plumbing smoke test); `_train_agentic_calc` (the shared tool-stage pipeline: optional SFT warm-up → RLCluster/GRPOLearner → train); `train_agentic_t0/t1/t2` (per-stage wrappers wiring the dataset builder, segment fn, system prompt, `env_max_steps`, and budgets). Also the metrics-capture classes and `_NormalizingGRPOLearner`. |
| `launch_agentic.py` | The **iris entrypoint** (env-var interface, `DELPHI_AGENT_MODE={port,t0,t1,t2}`). Downloads Delphi, dispatches to the right `train_agentic_*`, prints per-step metrics. |
| `delphi_qwen3.py` | `delphi_config()` (Delphi's exact Qwen3 dims) + `load_delphi()` (safetensors load with a **hard 124/124 key-coverage assertion**, applies the RoPE patch) + `load_tokenizer()` (Llama-3, pad=eos). Exports `DELPHI_BOS_ID` / `DELPHI_EOS_ID`. |
| `delphi_patch.py` | `patch_tunix_rope_for_delphi()` — the worker-shippable RoPE monkeypatch (bakes in `rope_theta=500000` + Llama-3 `rope_scaling`) for exact HF parity on stock `google-tunix 0.1.7`. |
| `arithmetic.py` | The **non-agentic** arithmetic/algebra env (`build_arithmetic_dataset`, `answer_reward` / `format_reward` / `proximity_reward`, `metric_fn`) used by `train_agentic_port` and the original feasibility runs. |
| `train_delphi.py` / `launch_delphi.py` | The original non-agentic GRPO harness + its iris entrypoint. |
| `toy_cats.py` / `launch.py` | The "emit more cats" toy GRPO smoke task + entrypoint. |
| `test_*.py`, `_validate_*.py`, `_probe_*.py` | Dev-time gates / validation scripts (HF parity, reward correctness, format probing). |
| `pyproject.toml`, `uv.lock` | The validated tunix-native manifest (no `marin-levanter`; see `DESIGN.md` §4). |

---

## Invariants & gotchas — DO NOT regress these

These are hard-won (see `REPORT.md` §9 for the full rationale). Each comes with a
guard already in the code; keep it.

### A. RL learns the tool *call*, not the result *copy*

Copying an injected `Tool result: X` (into the next call or the final answer) is
**out-of-distribution** for the base LM — sampled too rarely for GRPO to amplify
(plain T0 RL drove `arg_acc` → ~0.99 but `solve_ratio` peaked ~0.1 then
**collapsed**). The fix is a short **SFT warm-up** that makes call+copy
in-distribution *before* RL, on the **same in-memory nnx actor** (no checkpoint
round-trip — both phases mutate the same module and `RLCluster` re-shards it).
Guard: `run_sft_warmup` (`agentic_sft.py`), invoked when `sft_steps > 0` in
`_train_agentic_calc`. Never remove the warm-up from the tool stages.

### B. Gradient clipping is load-bearing — its absence is a *crash*

Unclipped multi-turn GRPO hits `inf`/`NaN` grads that surface as a **libtpu
`SIGSEGV`** (lr 2e-5 died ~step 3; lr 1e-5 ~step 99) — a crash that loses all
progress, not mere instability. `optax.chain(clip_by_global_norm(1.0),
adamw(...))` both eliminates the crash and stabilizes training. Guard:
`agentic_common.clipped_adamw` — the **single** source for the optimizer, used by
the SFT phase *and* the RL phase. Clip both; never swap in a bare `adamw`.

### C. SFT over-collapses if too strong

Too much warm-up sharpens the policy onto a degenerate continuation and drives
`tool_call_rate → 0`. The sweet spot is **~150–250 transcripts** (T0/T1 robust at
150; T2 wants 250); ~400 collapses T0. Warm-up is a *nudge*, not a fine-tune.
Guard: keep `DELPHI_SFT_STEPS` in that band; don't crank it to "more is better".

### D. Train/RL prompt mismatch silently breaks tool-call emission

SFT must use the **same prompt distribution** as the RL rollout. A naive warm-up
trained on `BOS Q:… → CALC` but RL prompts with `‹few-shot› Q:…` — a context the
warm-up never saw — so the model dropped the `CALC(` and emitted a bare
`92 * 98`. Benign for single-call T0; **corrupts chained T1/T2 turn-1**. Fix:
**prepend the masked few-shot prompt to every SFT transcript** (`prompt_prefix`
in `_SFTSource` / `build_sft_dataset`, passed as `sft_prompt_prefix=T{1,2}_SYSTEM_PROMPT`
from `train_agentic_t1/t2`). Generalizable lesson: warm-up transcripts must match
the RL prompt distribution, **prefix included**.

### Base-LM surface constraints (carried from the arithmetic work)

- **No chat template.** Use `DelphiRawTextChatParser` (raw text), never the stock
  Qwen template (it emits out-of-vocab `<|im_start|>` control strings).
- **Never emits EOS.** Each single-line turn must stop at its line break. Stop on
  a **digit/`)`-terminated newline** (`newline_terminal_eos_tokens`), plus the
  bare `"\n"`. A 198-only stop misses the fused `")\n"` / `"<digit>\n"` tokens and
  the model runs past the line. The chat parser's `generation_suffix="\n"` ensures
  the model emits content *before* the first newline (so the stop doesn't fire on
  an empty turn, which the engine would discard).
- **Can't reliably emit Qwen JSON.** The tool surface is bare-text `CALC(a * b)`,
  not `<tool_call>{…}</tool_call>`. The JSON surface measured `tool_call_rate ≈ 0`
  on TPU (operands right, `op`/braces mangled).
- **Llama-3 BPE fuses `)\n`.** A correctly-closed `CALC(a * b)` is recorded as
  `CALC(a * b` (the `")\n"` stop token is stripped). The parser, `arg_reward`, and
  the metrics treat the **closing `)` as optional** (`_CALC_PREFIX_RE`); don't
  tighten them to require it.
- **GRPO group must vary.** tunix 0.1.7 generates each group member with a fixed
  `RolloutConfig.seed`, so all `num_generations` samples would be byte-identical →
  zero advantage → no gradient. Guard: `install_per_call_rollout_seed`. Keep it.
- **`kv_cache_size ≥ max_prompt_length + max_tokens_to_generate`** or the sampler
  hard-errors (the drivers add `+ 8` headroom).
- **Actor storage must be fp32** (bf16 rounds small Adam updates to zero). Compute
  can be bf16 via `config.dtype`. The reference (if `beta > 0`) is bf16.

### CPU is for compile/import/unit checks ONLY

CPU validates `import` / compile / pure-logic units (parsers, segment encoding,
dataset shapes). **Never** validate a training objective or a learning claim on
CPU — all training objectives are validated on **TPU (v6e, free on iris)**. Use
TPU aggressively (parallel sweeps); use CPU only to catch breakage before paying
for a TPU.

---

## How to add a new curriculum stage

The stages differ **only in chain depth** (number of `*` operands), so adding a
deeper one (e.g. T3 = `a*b*c*d*e`, depth 4) is a small, parameterized change:

1. **Operand names** — `agentic_tools._OPERAND_NAMES` currently has `a,b,c,d`
   (covers depth ≤ 3, i.e. ≤ 4 operands). For depth 4 add `"e"` so the dataset can
   name the extra operand column.
2. **System prompt** — add a `T3_SYSTEM_PROMPT` few-shot demo in `agentic_tools.py`
   (two full worked transcripts with depth-4 chains; match the exact line format).
   These are hand-tuned literal strings — keep them byte-exact.
3. **Dataset / segments** — already parameterized: `build_chain_dataset(...,
   depth=4)` and `chain_segments(rng, depth=4)` just work; add thin
   `build_t3_dataset` / `t3_segments` wrappers if you want named entrypoints (the
   existing `t0/t1/t2` wrappers are the pattern).
4. **Driver** — add `train_agentic_t3` in `train_agentic.py` mirroring
   `train_agentic_t2`: pass `dataset_builder=build_t3_dataset`,
   `sft_segment_fn=t3_segments`, `system_prompt=T3_SYSTEM_PROMPT`,
   `sft_prompt_prefix=T3_SYSTEM_PROMPT` (invariant **D**), `env_max_steps=5`, and
   **bumped** `max_prompt_length` / `max_tokens_to_generate` / `sft_max_seq_len`
   (the chain is longer and the intermediates are bigger).
5. **Launcher** — add `"t3": train_agentic_t3` to the `_tool_train_fns` map and
   `"t3"` to the `DELPHI_AGENT_MODE` validation set in `launch_agentic.py`.
6. **Budget** — expect a slightly higher SFT budget for deeper chains (copies of
   longer numbers are marginally harder); start at `DELPHI_SFT_STEPS=250`.

## How to add a new tool

The calculator is the only tool today; a second tool touches the surface, the
parser, the agent's tool map, and the env reward:

1. **Tool** — reuse a stock `tunix.rl.agentic.tools` tool or write one (a class
   with the tool's execute contract); register it in the tool map (`T0_TOOL_MAP`
   pattern) under a name.
2. **Surface + parser** — pick a **bare-text** surface the base LM can emit (not
   JSON — invariant on Qwen JSON being OOD), and write a parser duck-typing
   `CalcTextToolParser` (`parse(text) -> [ToolCall]`, `get_tool_prompt -> ""`).
   Keep parsing lenient about the fused trailing newline / closing delimiter.
3. **Few-shot demos** — the base LM has no tool docs; the *demonstrations* carry
   the format. Add a system prompt with worked transcripts in the new surface.
4. **Agent** — if the tool's result needs a specific injected rendering, mirror
   `DelphiToolAgent._observation_to_messages` (render the raw result so
   `DelphiRawTextChatParser` produces the exact `Tool result: …` the demos show —
   a divergent prefix derails the base LM's continuation).
5. **Reward** — give a **dense** copy/solve signal (the sparse "final == gold"
   reward is too sparse for the base LM to bootstrap — see `CalcToolEnvironment`).
6. **SFT** — add a `*_segments` transcript builder teaching the new call+copy, and
   prepend the matching `sft_prompt_prefix` (invariant **D**).

---

## Style & verification

- **2-space indent**, Google-style docstrings. Keep the explanatory docstrings /
  comments that record *why* (the four findings, the BPE / EOS / seed gotchas) —
  condense if verbose, but don't strip the rationale.
- Refactors of the tool/SFT code must be **behavior-preserving**. Preserve the
  public entrypoints `train_agentic_port` / `train_agentic_t0/t1/t2` and the
  `DELPHI_AGENT_MODE` env-var interface; update **all** call sites if you rename
  or re-parameterize internal helpers (`grep` the dir).
- Verify before reporting: `uv run python -c "import agentic_common, agentic_sft,
  agentic_tools, train_agentic, launch_agentic"` must import cleanly, and exercise
  the CPU-runnable pure-logic units (`CalcTextToolParser.parse`,
  `parse_tool_call_operands`, `is_well_formed_tool_call`, `chain_segments` /
  `_encode_segments`, `DelphiRawTextChatParser.parse`, the dataset builders) to
  confirm outputs are unchanged. Do **not** run TPU jobs or `iris` from here — the
  orchestrator submits those.
