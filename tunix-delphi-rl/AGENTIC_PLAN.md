# Plan: from arithmetic GRPO to tool-calling and 3-step agentic RL (Delphi on tunix/iris)

Forward plan after the tunix-on-iris evaluation (weaver #229, PR #2). Grounded in a 7-agent research sweep of the tunix source, every load-bearing claim verified against the **stock `google-tunix==0.1.7` sdist** that runs on the iris worker (not just the newer clone).

---

## TL;DR

- **Feasible on what we already run.** tunix ships a complete agentic/tool-calling RL stack *inside the 0.1.7 wheel* (`tunix/rl/agentic/`: tool/model agents, `CalculatorTool`, a Qwen `<tool_call>` parser, multi-turn rollout, an agentic `GRPOLearner`). **Zero new dependencies**, **vanilla sampler unchanged**, and **~70–100% of our harness carries over** (model loader, rope fix, tokenizer, mesh, `RLCluster`, iris launch, metrics).
- **The change is small and surgical:** swap the non-agentic `GRPOLearner`/`GRPOConfig` for the agentic ones (a 2-import change), pass `agent_class`/`env_class`/`chat_parser`, and write **two genuinely new components**: (1) a **raw-text chat parser with a `tool`-role branch** (Delphi has no chat template; the stock Qwen parser emits out-of-vocab `<|im_start|>`), and (2) **env-closure rewards** (`reward_fn(task, action) -> float`, a different signature than our `(prompts, completions)` rewards).
- **The narrative writes itself.** Our headline finding was that Delphi *can't* learn wide-answer arithmetic internally (2-digit/multi-step → ~0% solve → zero GRPO advantage). **A calculator tool is the exact fix:** it collapses a wide answer space into "emit one call, copy the result." So the first agentic task is *motivated by the thing that failed*, not invented.
- **The real risk is learnability, not engineering:** can a 447M *base* LM emit a parseable `<tool_call>` from cold? Cold `tool_call_rate ≈ 0` reproduces our zero-variance wall in agentic form. Mitigated by few-shot tool transcripts + "used the tool" reward shaping + warm-starting from the arithmetic checkpoint — and **settled cheaply by a temperature-0 format probe before spending any GRPO compute.**
- **Smallest provably-valuable first step (~1 day):** single calculator call on 2-digit multiplication (`max_steps=2`) — the exact internal-failure case, now a copy-the-result task.
- **Full ladder: ~7–9 days** of focused work.

---

## 1. Why this is the right next step

Three reasons it's the natural progression and not a detour:

1. **It directly extends the proven result.** We have GRPO-on-iris working with Delphi end-to-end. Agentic GRPO reuses the *same* `RLCluster`, sampler, mesh, fp32 actor, and iris job — it is the smallest meaningful step up the capability ladder, not a rewrite.
2. **It attacks our open problem.** The answer-space-density wall (wide-answer arithmetic is unlearnable cold) is precisely what tool use solves. T0 turns a 0%-learnable task into a learnable one *without changing the model's arithmetic*, isolating the new skill (call a tool, read its output, act on it).
3. **It generalizes.** "Emit a structured action → read an observation → decide the next action" is the core agentic loop. Once it works for a calculator at horizon 1→3, the same machinery extends to any tool (lookup, compare, code-exec) and longer horizons.

---

## 2. What tunix gives us (verified in 0.1.7)

| Component | What it is | File (0.1.7) |
|---|---|---|
| Agentic `GRPOLearner` | Group-relative GRPO over multi-turn trajectories; takes `agent_class`, `env_class`, `chat_parser` | `tunix/rl/agentic/agentic_grpo_learner.py` |
| `ToolAgent` / `ModelAgent` | Policy wrappers; ToolAgent injects a tools prompt and parses tool calls | `…/agents/{tool,model}_agent.py` |
| `ToolEnvironment` | Runs tool calls, injects results as `role:"tool"` messages, ends on `finish`/string/`max_steps`; reward via `reward_fn(task, action)->float` | `…/environments/tool_environment.py` |
| `CalculatorTool` | One binary op `a OP b`, `op∈{+,−,*,/}`, returns `str(result)` | `…/tools/calculator_tool.py` |
| `QwenToolParser` | Pure string-scan for `<tool_call>{json}</tool_call>` (no special tokens needed) | `…/parser/tool_parser/qwen_parser.py` |
| Rollout engine | Async producer/consumer; one sampler call **per turn** | `…/trajectory/trajectory_collect_engine.py`, `…/pipeline/rollout_orchestrator.py` |

**Reward & advantage semantics:** still GRPO, group-relative across the `num_generations` rollouts, computed **per-trajectory (episode-level)** — there is *no* per-turn credit assignment in the loss, so all "stepwise credit" lives in the env `reward_fn` we write. Multi-turn rollouts are flattened to one `(prompt, completion)` token sequence with an assistant-vs-environment **token mask** (only model tokens get loss).

**Two hard config facts (0.1.7 validates them):** `max_tokens_to_generate == max_response_length`, and `use_rollout_logps=True` requires `return_logprobs=True`. We set **`use_rollout_logps=False`** (trainer recomputes logprobs) — what the GSM8K demo does, and it sidesteps a 0.1.7 cross-turn token-alignment weakness (see Risks R2).

**Version-drift trap:** the local clone (`9beef62`) and 0.1.7 are *divergent branches*, not linear. `trajectory_collect_engine.py` differs (e.g. `_run_with_timing` arity; clone-only `update_assistant_end_tokens`). **Write against the installed 0.1.7 API; do not backport clone engine code to the worker** — it will crash on the signature mismatch.

---

## 3. The two new components we must build

1. **`DelphiRawTextChatParser`** (~20–40 lines). Models the GSM8K demo's `VTCRawTextParser` (`examples/agentic/qwen3_grpo_gsm8k_demo.py:461`) **but adds a `role:"tool"` branch** the demo lacks — without it, tool results are silently dropped from the model's next-turn context *and* the training stream, making the task unlearnable with no error. Renders turns as plain text with legible role markers (`User:` / `Assistant:` / `Tool:`) joined by `\n`; sets the stop token to Delphi's eos (`128001`). This is the single most load-bearing new piece.
2. **`CalcToolEnvironment(ToolEnvironment)`** (~15–40 lines) + **env reward closures**. Subclass to record per-episode tool facts (`(name, args, result)`, whether a non-`finish` call ran, whether any tool errored) so the reward closure can read them; re-express each stage's reward as `reward_fn(task, action) -> float`. Unit-test every closure on hand-built trajectories before any TPU run (reward-signature foot-gun, R7).

Everything else is configuration.

---

## 4. Milestone ladder

Ordered by **horizon length** (the true difficulty axis for an agent — one malformed `<tool_call>` ends the episode), front-loaded with two cheap de-risking milestones the naive "1→2→3 call" ladder omits.

| # | Milestone | Proves / de-risks | Effort | Advance gate |
|---|---|---|---|---|
| **M-port** | Single-turn agentic plumbing, **no tools**: agentic `GRPOLearner` + default `ModelAgent`/`TaskEnvironment` + our raw-text parser, running our *known-learnable* single-digit-add dataset. | Isolates the learner/config swap + parser from all tool/multi-turn risk. Reproduces our 6%→65% add curve through the agentic path. | 0.5–1 d | Reproduce add learning; `logp_diff` canary clean |
| **M-format** | Teach the `<tool_call>` syntax cheaply *before* outcome RL: few-shot tool transcripts + a format-only reward; **temp-0 probe** (reuse `_probe_arith_format.py`) of whether Delphi can imitate the format at all. | Kills the #1 failure (cold `tool_call_rate≈0` → all-zero groups). Agentic analogue of our few-shot prefixes. | 0.5–1 d | `tool_call_rate > 0` on cold eval |
| **T0** | **Single tool call:** 2-digit multiply → 1 calculator call → finish. `max_steps=2`. Warm-start from M-format / arithmetic checkpoint. | The core skill: emit one valid call, copy the narrow result. Converts a 0%-learnable task into a learnable one. | 1–1.5 d | `answer_solve ≥ 70%`, `tool_call_rate ≥ 0.9`, `copy_acc ≥ 0.9` |
| **T1** | **Two tool calls:** `(a OP b) OP c`, `max_steps=3`, second call's operand = first call's result. Warm-start from T0. | First genuine multi-turn coherence (read a tool result, feed it back). | 1 d | `answer_solve ≥ 50%`, `two_call_rate ≥ 0.8`, `chain_acc ≥ 0.7` |
| **T2** | **3-step agentic.** Build the **3-hop single-tool calculator word problem first** (reuses everything), then upgrade to a **compute-then-compare** task with a 2nd `compare` tool (genuine state-dependent branch). `max_steps=3`. Warm-start from T1. | The decision skill: choose the next action based on an observed intermediate result. | 1.5–2 d (+1 d upgrade) | `full_solve ≥ 35%`, `branch_ok ≥ 0.6`, `compute_ok ≥ 0.8` |

**Total ≈ 7–9 days.** Two sequencing rules from the critics: **M-port is non-negotiable as step 1** (so a failure is diagnosable as "learner swap" vs "tools"), and **T2 leads with the 3-hop single-tool version** (the compute-then-compare multi-tool variant adds wiring risk and goes second).

### Per-stage reward sketches (env `reward_fn(task, action) -> float`)

All keep exact-correct = the dominant term so `solve_ratio` stays a clean exact-match metric, with dense shaping terms to guarantee non-zero within-group variance from cold (the lesson from the density finding).

- **T0:** `r = 1.0·[pred==gold] + 0.2·[valid calc call ran] − 0.2·[any tool errored]`. The `+0.2` "used the tool" term is what gives cold groups variance *before* any answer is right.
- **T1:** `r = 1.0·[pred==gold] + 0.15·[exactly 2 calls] + 0.25·[2nd operand == 1st result] − 0.2·[errored]`. The chain term rewards correctly threading the intermediate even when the final is wrong.
- **T2 (compute-then-compare):** `r = 0.2·compute_ok + 0.2·compare_ok + 0.2·branch_ok + 0.4·value_ok − 0.2·errored`; "solved" at `r ≥ 0.95`. `branch_ok` rewards making the decision the observation implies — the skill T2 tests.

---

## 5. Curriculum principle (one idea)

**Keep within-group reward variance non-zero from the very first rollout.** GRPO learns only from groups with mixed outcomes; a task the cold policy never (even partially) succeeds at yields identically-zero advantage. Everything serves this:
1. **Few-shot the tool format** (1–3 fully-worked `<tool_call>`/tool-result/`finish` transcripts in the prompt) so cold rollouts sometimes parse — the agentic analogue of our working few-shot arithmetic prefixes.
2. **Order by horizon, not arithmetic difficulty** (the tool already trivialized the arithmetic).
3. **Warm-start each stage from the previous checkpoint**, and start T0 from the **arithmetic-trained Delphi** (it already reliably emits a single integer after a marker — the exact "copy the result" sub-skill).
4. **Tune the operand distribution** so the tool is *necessary* (internal solve ≈ 0 → forces tool use) but the result is *short* (copy stays reliable).
5. **Promote on a variance gate**, not just a solve gate: advance only when the next stage's cold eval already shows `tool_call_rate > 0`.

---

## 6. Reuse map (what we keep vs. build)

- **Carries verbatim:** `delphi_qwen3.load_delphi` + `delphi_patch` rope fix, tokenizer (pad=eos, `eos_tokens=[128001]`), fp32 actor / bf16 ref, `_build_mesh` + `role_to_mesh`, `RLCluster(...)`, `rollout_engine="vanilla"`, `RLTrainingConfig`, metrics hook, the env-driven `launch_delphi.py` skeleton.
- **2-import swap:** `tunix.rl.grpo.grpo_learner` → `tunix.rl.agentic.agentic_grpo_learner` (`GRPOLearner` + `GRPOConfig`); set `use_rollout_logps=False`, `max_tokens_to_generate == max_response_length`.
- **Reused for M-port/M-format, rewritten for tool stages:** `arithmetic.py` dataset (same grain shape; `answer` becomes the env `task`) and rewards (usable as `reward_fns=` single-turn; rewritten as env closures for tools).
- **New builds:** the raw-text chat parser (mandatory), `CalcToolEnvironment` + reward closures, and (T2 upgrade) a `compare` `BaseTool`.

---

## 7. Risk register

| # | Risk | Mitigation |
|---|---|---|
| R1 | **Cold `tool_call_rate ≈ 0`** (base LM never saw `<tool_call>`) — the density wall in agentic form | Few-shot tool transcripts; `+0.2`/`−0.2` tool shaping; M-format + temp-0 probe *before* GRPO; fallback to a simpler tool surface syntax (e.g. `CALC(47*53)`) + ~20-line custom parser if JSON imitation fails |
| R2 | **No chat template** (stock Qwen parser emits out-of-vocab `<|im_start|>`) | Custom raw-text parser; `use_rollout_logps=False`; watch the `logp_diff` canary (engine logs it) |
| R3 | **Multi-turn incoherence** at 447M (forgets tool result, finishes early) | Tiny `max_steps` (2–3); per-step partial credit; warm-start; short tool outputs |
| R4 | **Context growth** (vanilla sampler re-encodes each turn, no cross-turn KV reuse) | ≤3 few-shot demos; `max_steps ≤ 3`; size `kv_cache_size` for cumulative context; modest `max_concurrency` |
| R5 | **Internal-solve leakage** (answers without the tool) | Operands wide enough that internal solve ≈ 0; `tool_call_rate` + `copy_acc` gates; push to 3-digit if leaking |
| R6 | **Zero-variance groups stall** | ≥3 partial-credit tiers per reward; variance-gated promotion; `degenerate_group_masking=False` to *see* dead groups |
| R7 | **Reward-signature mismatch** (`(prompts,completions)` vs env `(task,action)`) | Rewrite as env closures; unit-test on hand-built trajectories before TPU |

---

## 8. Infra / operational notes

- **Job shape unchanged:** one v6e-4, colocated roles, same iris submit line. Only *per-step wall-clock* and *host RAM* grow with turn count (1 sampler call **per turn**, no KV reuse) — keep horizons tiny.
- **No new packages.** The tool stack is stdlib-only (`json`, `ThreadPoolExecutor`); no sandbox/subprocess/Docker. (deepswe's `r2egym` and frozenlake's `gymnasium` are *not* iris-portable and are out of scope.)
- **Recipes aren't shipped in 0.1.7** (`examples/` is absent from the wheel) — copy the `VTCRawTextParser` pattern into our harness rather than import it.
- **Extend `launch_delphi.py`** with `DELPHI_AGENT_MODE`, `DELPHI_MAX_STEPS`, `DELPHI_TOOL` env knobs; keep the download+rope+run skeleton.

---

## 9. Open decisions for the user

1. **Scope:** stop at T0 (prove base-model tool-calling — the cleanest standalone result), go through T1, or commit to the full T2 ladder?
2. **Tool-call syntax fallback:** if the temp-0 probe shows Delphi can't imitate Qwen `<tool_call>` JSON cold, do we accept a simpler custom surface syntax (`CALC(47*53)`) + a tiny custom parser? (Recommended yes — far more base-LM-friendly, ~20 lines.)
3. **SFT warm-up:** allow a tiny supervised warm-start on synthetic tool transcripts before RL if M-format's few-shot alone isn't enough? (DeepSeek-R1-style cold-start; cheap insurance against R1.)
4. **Where this lives:** new branch/PR for the agentic work, or extend the current one?

**Recommended first action:** implement M-port + the raw-text parser and run the temp-0 `<tool_call>` format probe — together they settle the only real uncertainty (base-model tool-call feasibility) in ~1 day before any larger commitment.

---

*Sources: tunix 0.1.7 sdist + clone @ `9beef62`; ToRL (arXiv:2503.23383), Hard Examples Are All You Need (arXiv:2508.14094), Zylos tool-use-RL reward design (2026), Scaf-GRPO/GHPO scaffolding. Full per-agent research dossiers in the workflow transcript.*
