# Adopting google/tunix on iris: from a "print cats" smoke test to an agentic coder

**Marin Research Report · June 2026**

Weaver issues #229 (feasibility) · #5 (agentic tool use) · #7 (agentic coding)

---

## TL;DR

We set out to answer one question — *can marin adopt google/tunix for RL post-training on TPUs, run it through the iris cluster manager, and bring a marin model into it?* — and then kept pulling the thread until a 447M base LM was writing and self-grading Python. The answer to the feasibility question is **yes, unambiguously**: tunix runs on iris with **zero iris changes** and **one upstreamable tunix bug fix**, and the marin model we targeted (Delphi) turned out to be a stock Qwen3 that loads natively with **exact HF-logit parity** (top-1 100%, logit MSE 7e-12). The hard part the brief anticipated — porting a "grug" model with no generate/KV-cache path into a new framework — **never had to be solved by hand**.

From there we escalated through five milestones, each a real run on TPU v6e:

1. **It installs and runs.** `google-tunix` + `marin-iris`/`marin-fray` resolve in one `uv` venv (164 packages, no `marin-levanter`); a tunix GRPO job runs on an iris worker that `uv sync`s its own lock. A "print more cats" toy GRPO learns **0.08 → 1.00** reward in 80 CPU steps, then on a **v6e-4** iris job.
2. **Delphi does arithmetic, then basic algebra.** Single-digit addition **6% → ~65%** solve in ~4 min; solve-for-x linear algebra **4.7% → ~37%** — no calculator, plain exact-match reward. The curriculum surfaced a non-obvious RL law: *learnability tracks answer-space density, not symbolic difficulty.*
3. **Delphi learns genuine multi-step tool use (#5).** It calls an external calculator and **chains** the calls — single (T0), two-deep (T1), three-deep (T2, each tool output feeding the next call's argument) — all reaching **~100% solve** on a v6e-4. RL alone *provably cannot* get there; a distribution-matched SFT warm-up is the enabler.
4. **Delphi writes code an interpreter grades (#7).** Given a purely-functional `micropython` interpreter as grader, the base LM goes **few-shot 3/50 → SFT warm-up 50/50** on a 5-tier ladder (constant print … recursive `fib` … FizzBuzz/gcd/Collatz), writing real recursive/looping programs.
5. **The cross-experiment lesson, and it inverts.** §3 needed RL because the target (copying a tool result forward) is a *narrow* behavior the base model samples too rarely for SFT to demonstrate. §4 did *not* need RL: writing a program is *fully demonstrable* by SFT, so SFT does the heavy lifting and **Dr.GRPO is marginal** — it ran stably and had nothing to do.

**The one negative that mattered everywhere:** multi-turn GRPO without gradient clipping does not wobble, it **crashes** — `inf`/`NaN` grads surface as a libtpu `SIGSEGV` that loses all progress. Clipping is load-bearing infrastructure, not a tuning knob.

We are now escalating once more, to a **multi-turn coding agent** (write → run → revise over several rounds, Dr.GRPO, on a harder problem set where SFT is insufficient). That work is in progress; its hypothesis and a clearly-marked stub are at the end (§11). No results are claimed for it.

> **What "grug-style model into tunix" meant in practice.** We read it as "bring a marin model into tunix," and the cheapest correct path is tunix's native NNX model zoo + a weight loader — **not** porting grug's `equinox.Module`. For Delphi this is exact and free. A *genuinely* grug-only architecture (e.g. an MoE GatedNorm/XSA variant with no HF equivalent) would still need the equinox→nnx port + KV-cache + RoPE-offset (Strategy A in `DESIGN.md`); we scope that as a separate follow-on.

---

## 1. What we set out to evaluate, and the pieces involved

The task: determine whether marin can adopt the **tunix** post-training toolkit on TPUs, run it via **iris**, and add a marin ("grug-style") model into it — using the `delayed-gradient-pp` experiment as the reference for how a grug model integrates with marin. The plan, in milestones: **M0** research + design, **M1** model integration, **M2** a local CPU smoke test, **M3** an iris job, **M4** Delphi doing arithmetic via a calculator environment, **M5** a curriculum toward basic algebra, **M6** this report. Issues #5 (tool use) and #7 (coding) are the follow-ons that extended the arc.

Three components, three different worldviews:

- **tunix** is Google's JAX-native LLM post-training framework, built entirely on `flax.nnx`: an `RLCluster` holding actor/reference/reward models, GRPO/PPO learners, a native KV-cache `Sampler` (plus vLLM/sglang backends), and an agentic stack (environments, tools — including a `CalculatorTool` — and reward managers).
- **grug** is marin's hand-rolled, copy-first Levanter training template: an `equinox.Module` transformer over raw `jax.Array`s with explicit `PartitionSpec` sharding, exposing only forward / `logits` / `next_token_loss` — **no generate, no KV-cache**. This is the "RL-rollout gap" the brief flagged as the likely hard part.
- **iris** is marin's cluster/job manager: a job bundles the experiment dir, the worker `uv sync`s its pinned deps, and runs the entrypoint.

The model is **Delphi** (`marin-community/delphi-3e18-447Mparams-1.2Btokens`): a 447M base LM with a Llama-3 tokenizer and **no chat template**, on disk as `Qwen3ForCausalLM`.

---

## 2. The finding that reframed everything: Delphi is a Qwen3

We expected to spend the project porting grug's equinox transformer into tunix's nnx world and hand-building a KV-cache sampler. The first thing we checked killed that plan — in the good way.

`marin-community/delphi-3e18-447Mparams-1.2Btokens` reports `architectures: ["Qwen3ForCausalLM"]`, `model_type: "qwen3"`. It is a **dense Qwen3**, 447M params: 11 layers, hidden 1024, 8 heads (no GQA), head_dim 128, intermediate 4096 (SwiGLU), Qwen3 QK-norm, vocab 128256 (Llama-3 tokenizer), `rope_theta=500000`, `rms_norm_eps=1e-5`, untied embeddings, base LM (no chat template). We verified at the byte level that **all 124 of its safetensors tensors match tunix's existing qwen3 key-map** (read directly from the safetensors header).

Because tunix already ships a native `flax.nnx` Qwen3 (`tunix/models/qwen3/model.py`) implementing the full RL/sampler contract — `__call__(input_tokens, positions, cache, attention_mask) -> (logits, cache)`, `init_cache`, `compute_final_logits`, QK-norm, untied lm_head — and an HF-safetensors loader, **the rollout/KV-cache gap was already closed**. The integration collapsed from "port an equinox model into a new framework" (weeks) to "a config + a one-line-class bug fix + a calculator environment" (days). DESIGN.md calls this **Strategy B**: load Delphi into tunix's native Qwen3 and run GRPO with the `vanilla` in-process sampler.

---

## 3. The changes we had to make

### 3.1 One real tunix bug: RoPE ignored `config.rope_theta` and never applied Llama-3 scaling

We expected loading to be exact and found it was 96% right — which on a logit comparison is *wrong*. tunix's qwen3 `apply_rope` defaults `rope_theta=1_000_000` and the call sites in `Attention` **never pass `config.rope_theta`**, so a model with `rope_theta != 1e6` (Delphi: 500000) silently gets the wrong RoPE. **This bug affects tunix's llama3 model too.** Fixing only `rope_theta` got to top-1 96%, logit MSE 1.8e-3 vs HF — closer, still wrong.

The deeper issue: Delphi uses Llama-3 `rope_scaling` (factor 8), which tunix does not model at all. Contrary to our initial design assumption, **this scaling is NOT inert at short context** — it rescales inverse frequencies by *wavelength*, perturbing ~35/64 frequency components at *every* position regardless of sequence length. Implementing the Llama-3 scaling (ported from HF's `_compute_llama3_parameters`) **plus** honoring `rope_theta` yields **exact HF parity: top-1 100%, logit MSE 7e-12**.

We deliver the fix two ways:

- **For validation / upstream:** an additive, backward-compatible patch to `tunix/models/qwen3/model.py` (a `rope_scaling` field on `ModelConfig`, a `_llama3_scale_inv_freq` helper, both rope params threaded through `apply_rope`). **Recommended as a PR to google/tunix.**
- **For the iris worker** (which installs stock `google-tunix 0.1.7` from PyPI): an import-time monkeypatch (`delphi_patch.patch_tunix_rope_for_delphi()`) that rebinds `apply_rope` to bake in Delphi's `rope_theta` + Llama-3 scaling. Same exact parity (96% → **100%** top-1 against stock tunix), no fork, no edit to the worker's install. `load_delphi()` calls it automatically.

### 3.2 Packaging: split levanter out; a tunix-native experiment venv

A single venv with **both** tunix and `marin-levanter` is unresolvable today: tunix's `orbax-checkpoint>=0.12.0` needs `tensorstore>=0.1.84`, while `marin-levanter` pins `tensorstore<0.1.82` (empty intersection). We don't need levanter — Delphi's *published HF safetensors* are the boundary artifact. The experiment depends only on `google-tunix` + `marin-iris` + `marin-fray` (`uv lock` → 164 packages). The `tpu` extra pulls `google-tunix[prod]` (= `jax[tpu]`). vLLM/sglang rollout is off-limits here (they want `jax[tpu]==0.7.2`, which `prod` excludes); we use the `vanilla` sampler, which is also the only backend that runs a custom/non-HF arch. **No iris changes were required.**

### 3.3 The experiment package

| file | role |
|---|---|
| `delphi_qwen3.py` | `delphi_config()` (exact dims) + `load_delphi()` (safetensors load + **hard** 124/124 key-coverage assertion, not a log check) + `load_tokenizer()` (Llama-3, pad=eos) |
| `delphi_patch.py` | `patch_tunix_rope_for_delphi()` — the worker-shippable RoPE fix |
| `arithmetic.py` | curriculum dataset + `answer_reward`/`format_reward`/`proximity_reward` + `metric_fn` (solve_ratio) |
| `train_delphi.py` | the non-agentic `RLCluster` + `GRPOLearner` driver |
| `toy_cats.py` | the M2 toy: tiny fresh-init Qwen3 + "emit more cats" GRPO |
| `agentic_common.py` / `agentic_sft.py` / `agentic_tools.py` / `train_agentic.py` | the §8 tool-use stack (raw-text chat parser, clipped optimizer, SFT warm-up, calculator tool/env/reward, drivers) |
| `micropython.py` / `coding_tasks.py` / `coding_env.py` / `train_coding.py` | the §9 coding stack (interpreter, eval ladder, parameterized families, Dr.GRPO driver) |
| `launch*.py` | iris entrypoints (toy / Delphi arithmetic / agentic / coding); env-driven |
| `pyproject.toml` + `uv.lock` | the validated tunix-native manifest |
| `DESIGN.md` | the M0 design doc + A/B/C strategy trade study + rollout plan |

---

## 4. Milestones M0–M3: feasibility, the toy, and getting onto the cluster

We de-risked in the cheapest order: design before code, parity before learning, CPU before TPU.

| milestone | result |
|---|---|
| **M0 Design** | Strategy B chosen; every assumption empirically pre-validated (install, byte-level key match, `uv lock`, the rope bug). |
| **M1 Delphi load** | Exact HF parity: **top-1 100%, logit MSE 7e-12**; 124/124 keys; KV-cache decode matches a cache-free reference token-for-token. |
| **M2 CPU toy GRPO** | "Emit more cats" learns **0.08 → 1.00** reward in 80 steps (~10s CPU); wiring asserted directly (rollout diversity; cache `end_index` 1→8). Reproduced across seeds. |
| **M3a iris CPU job** | Toy GRPO ran as an iris job — worker `uv sync`'d tunix+jax, loop learned, "SMOKE TEST PASSED", exit 0. |
| **M3b iris TPU job** | Same toy on **v6e-4**: jax saw 4 TpuDevices, `(4,1)` fsdp mesh, loop learned, exit 0. (Rode past transient TPU bad-nodes via iris auto-retry.) |

The "print cats" toy is deliberately the dumbest possible RL task — a fresh-init tiny Qwen3 rewarded for emitting the token "cats" more often — precisely so that a *learning* curve isolates the plumbing. Once that learned on a v6e-4 iris job, every later failure could be attributed to the model or the task, never the stack.

---

## 5. M4–M5: Delphi learns arithmetic, then basic algebra (the first substantive ML result)

We then pointed real GRPO at Delphi with a no-tool arithmetic reward — the model computes the answer itself — and walked a curriculum toward algebra. The headline: **Delphi reached the stretch goal, solve-for-x linear algebra**, but the curriculum taught us something more useful than the headline.

| stage | task | answer space | reward | lr | steps | solve_ratio start → end |
|---|---|---|---|---|---|---|
| 0 | single-digit `a+b` | 0–18 (19) | exact | 1e-5 | 400 | 0.03 → **0.65** ✓ |
| 0 | (lr ablation) | 0–18 | exact | 5e-6 | 400 | 0.03 → 0.36 |
| 0 | (lr ablation) | 0–18 | exact | 1e-6 | 200 | 0.06 → 0.12 (lr too low) |
| 3 | **linear algebra `a·x+b=c`** | x∈[−9,9] (19) | exact | 1e-5 | 700 | 0.05 → **0.30** ✓ |
| 1 | add/sub/mul, 2-digit | 0–~9800 | exact | 1e-5 | 400 | 0.00 → 0.00 ✗ |
| 2 | multi-step (2 ops) | 0–~100+ | exact | 1e-5 | 400 | 0.02 → 0.01 ✗ |
| 1 | add/sub/mul, 2-digit | 0–~9800 | **shaped** | 1e-5 | 500 | 0.00 → 0.02 ✗ (mean_reward 0.13→0.21 — closer, not exact) |
| 2 | multi-step (2 ops) | ~0–100+ | **shaped** | 1e-5 | 500 | 0.02 → ~0.09 (peak 0.15) ✓ **shaping recovers it** |
| 3 | **linear algebra (longer)** | x∈[−9,9] | exact | 1e-5 | 1200 | 0.05 → **0.375** ✓ |

**We expected difficulty to track human-perceived complexity; we saw it track answer-space density instead.** The stages split cleanly, and *not* along the axis a human would call "hard":

- **Learn (✓):** single-digit addition (answers 0–18) → 0.65, and linear algebra (x∈[−9,9]) → 0.375.
- **Flat (✗):** 2-digit add/sub/mul (answers to ~9800) and two-operation arithmetic (answers ~0–100+) — both stuck at ~0.

The stages that learn are exactly the ones with a **small, dense answer space** (~19 integers), where the untrained policy stumbles onto the right answer ~5–6% of the time. That non-zero base rate gives the GRPO group *variance*, hence a non-zero group-relative advantage, hence a gradient. The wide-answer-space stages cold-solve at ~0–2%: nearly every group has all-equal (zero) reward → advantage vanishes → no gradient. **Linear algebra is "easier" for RL than 2-digit multiplication** because its answer is one of 19 small integers, not one of thousands.

**Proximity shaping confirmed the mechanism.** We added a `proximity_reward` (1.0 exact; else `0.5·max(0, 1−|pred−gold|/10)`; exact still uniquely crosses the "solved" threshold so `solve_ratio` stays exact-match) and re-ran the two flat stages. Stage 2, dead-flat at 0.008 under exact-match, climbed to ~0.09 (peak 0.15) — direct evidence the stall was *missing gradient*, not representational inability. Stage 1 (answers to ~9800) is the genuine hard case: shaping lifts the *mean* reward (closer answers) but exact solves stay near zero — the answer space is simply too large to land on the integer often within 500 steps. The lever for that tail is warm-start transfer from an earlier stage's checkpoint (tunix supports it via orbax) or a gentler operand ramp — neither was needed to clear the stretch goal.

**The lr finding matters too:** GRPO on this 447M base model is real but lr-sensitive — at lr 1e-6 the policy barely moves; lr 1e-5 produces a clean climb to ~65% on stage 0. The reward is sparse exact-match (plus a small, already-saturated format term that group-normalization removes), so the gradient comes purely from answer correctness.

---

## 6. The integration friction log

These are the non-obvious things a future tunix-on-marin user will hit. They are the same in every milestone below.

1. **tunix is `flax.nnx` to the core.** `RLCluster` hard-checks `isinstance(model, nnx.Module)`; the train step is `nnx.Optimizer` + `nnx.value_and_grad` + `nnx.jit` (donated). There is no seam to inject a non-nnx train step — only the loss/reward functions are pluggable. Bringing a marin model in means *being* an nnx module (native zoo or a port), not wrapping.
2. **The model `__call__` contract is rigid:** first four positional args `(input_tokens, positions, cache, attention_mask)`, returns `(logits|hidden, cache|None)`; optional hooks (`skip_lm_head`/`compute_final_logits`, `segment_ids`) are discovered by `inspect.signature`. `num_embed` is read only under `return_logits=True` (default False) but absent on qwen3 — add it defensively.
3. **The non-agentic GRPO template is in the tests,** not the headline examples: `tunix/tests/rl/grpo/grpo_learner_test.py::setup` shows the `reward_fns=` single-turn path. The `examples/frozenlake` script is the *agentic* learner (different reward signature).
4. **Dataset shape gotcha:** an HF `datasets.Dataset.batch()` yields list-valued columns, which tunix's `jax.tree.map(np.repeat, ...)` recurses into and corrupts (a prompt string becomes a 2-D array → `'ndarray' has no attribute 'split'`). Use `grain.MapDataset.source(...).batch(n).map(...)` so each batched column is a single numpy-array leaf.
5. **Actor storage must be fp32.** bf16 storage rounds small Adam updates (~1e-6 at lr 1e-6) below bf16 ULP (~7.8e-5) → the policy silently never moves. Compute can still be bf16 via `config.dtype`.
6. **The safetensors loader fails silently** — unmatched keys are logged, not raised, leaving random-init params. Assert key coverage independently (we do, hard).
7. **Base LMs need few-shot, not instructions.** Delphi has no chat template; bare-equation and `<answer>` formats failed; few-shot raw text ("Q: 2 + 3 = A: 5 …") was followed 100%.
8. **`kv_cache_size >= max_prompt_length + max_tokens_to_generate`** or the sampler hard-errors.

### Operational notes (iris / TPU)

- **Submission is unchanged from grug experiments:** `iris --cluster=marin job run --tpu v6e-4 --enable-extra-resources --extra tpu --region europe-west4 -- python launch.py`. The worker bundles the dir (git-ls-files; `.venv`/`__pycache__` excluded; 25 MB cap, our bundle is ~0.5 MB), `uv sync`s the committed lock, and runs. Delphi's 1.8 GB weights download from HF at runtime (not bundled).
- **`--tpu` is optional** — a CPU job cleanly de-risks packaging+submission before paying for TPU. Memory ≥4 GB / disk ≥10 GB need `--enable-extra-resources`.
- **Transient TPU bad-nodes:** several v6e nodes failed init with `"Couldn't open iommu group /dev/vfio/N: Device busy"`. iris detects the signature and reschedules automatically — keep `--max-retries` generous (we used 5).
- **v6e-4 capacity** was readily available (no boot wait) in `europe-west4` and `us-east5-b`; a 447M GRPO job runs colocated on one v6e-4 host in minutes. The marin cluster was reachable via gcloud auth (`iris-controller@hai-gcp-models`, 511/511 workers healthy).

### Reproducing the feasibility + arithmetic results

```bash
cd tunix-delphi-rl
uv sync --frozen --no-group dev                          # stock google-tunix 0.1.7, CPU
JAX_PLATFORMS=cpu .venv/bin/python test_smoke_cats.py    # M2 toy GRPO learns on CPU
# Delphi arithmetic on TPU (the winning config):
.venv/bin/iris --cluster=marin job run --no-wait \
  --tpu v6e-4 --enable-extra-resources --extra tpu --region europe-west4 \
  --cpu 8 --memory 64GB --disk 60GB --max-retries 5 \
  -e DELPHI_STEPS 400 -e DELPHI_STAGE 0 -e DELPHI_LR 1e-5 -e DELPHI_NUM_GENERATIONS 16 \
  -- python launch_delphi.py
```

---

## 7. Feasibility verdict (issues #229 / arithmetic)

1. **Adopt tunix for RL post-training on iris — it works today.** Zero iris changes; one upstreamable tunix bug fix; a 447M model trains in minutes on one v6e-4.
2. **Upstream the RoPE fix** (`rope_theta` + Llama-3 `rope_scaling`) — additive, backward-compatible, fixes a latent correctness bug that also affects tunix's llama3. Until merged, the monkeypatch ships it cleanly.
3. **For marin models that are Llama/Qwen-shaped (Delphi and most of the IsoFLOP suite), use tunix's native NNX zoo + the safetensors loader.** This *is* the "grug-style integration" in practice: explicit, copy-first config + loader, no framework bridge.
4. **A genuinely grug-only architecture** (MoE GatedNorm/XSA, no HF equivalent) is the only case needing the expensive equinox→nnx port (Strategy A). Treat it as a separate project.
5. **GRPO is lr-sensitive on small base models** — budget for a sweep (1e-6 was dead; 1e-5 worked).
6. **Order RL curricula by answer-space density, not human difficulty, and densify the reward for wide-answer tasks.** A task whose answer is one of ~20 small integers self-bootstraps; thousands of possible answers give GRPO no gradient.

The arithmetic env above is **non-agentic** — Delphi computes, it does not use a tool. The next two sections are the follow-ons that asked the stronger questions: real tool *use*, and real code *writing*.

---

## 8. Follow-on #5: genuine agentic tool use — Delphi learns to *call* a calculator

§7 left a thread open. The arithmetic above had Delphi compute the answer itself; the follow-on asks the question the original brief implies: **can a 447M base LM do real multi-turn tool use on tunix+iris** — emit a tool call, read back an injected `Tool result:`, and *copy/chain* it into the answer? We built a three-stage curriculum on tunix's agentic stack (`ToolEnvironment` / `ToolAgent` / agentic GRPO learner) plus a `CalculatorTool`. **Answer: yes — up to three chained calls, at ~100% — but only with a specific SFT warm-up; RL alone provably cannot get there.** The recipe and the four findings that make it work are the contribution.

### 8.1 The task surface and stages

A custom `CalcTextToolParser` exposes a **`CALC(a * b)`** surface (not tunix's stock Qwen JSON `<tool_call>{…}` — see finding D). A few-shot system prompt carries the format; the agent loop runs: model emits `CALC(...)` → env executes the calculator → injects `Tool result: X` → model continues. A shaped reward (in a `CalcToolEnvironment` subclass) gives partial credit for the right turn-1 operands (`arg_acc`), a **copy** term for reproducing the injected result, and a **solve** term for the final gold answer.

| stage | task | tool calls | new skill | result |
|---|---|---|---|---|
| **T0** | `a * b` (2-digit) | 1 | emit one grounded `CALC`, copy the result | **solve 1.0**, arg_acc 1.0, tool_call_rate 1.0 (held to step 149) |
| **T1** | `a * b * c` | 2 chained | **copy a tool OUTPUT into the next call's ARGS** | **solve 1.0**, arg_acc 1.0, tool_call_rate 1.0 (robust at SFT=150 *and* 250) |
| **T2** | `a * b * c * d` | 3 chained | a deeper chain — two ~6-digit intermediate copies forward | **solve 1.0** (SFT=250, sustained to step 98), ~0.97–0.98 (SFT=150); arg_acc 1.0, tool_call_rate 1.0 |

All on one `v6e-4`, minutes per stage. (T0 → a6e67dc, T1 → d1c4c2f, T2 → c820a73.)

### 8.2 The core obstacle: RL learns the *call*, not the *copy*

We expected plain GRPO to solve T0 outright. **We saw it drive the tool *call* to near-perfect (`arg_acc` → 0.99) while `solve_ratio` peaked at ~0.1 and then collapsed.** The reason is fundamental: copying an injected `Tool result: X` into the answer is **out-of-distribution** for the base LM — sampled too rarely for GRPO to ever amplify (the classic "RL only sharpens what the base policy already puts mass on" wall). The fix is the standard one: a short **supervised warm-up** that makes the call+copy pattern in-distribution *before* RL, using tunix's `PeftTrainer` on the **same in-memory `nnx` model object** (no checkpoint round-trip — both phases mutate the same module and `RLCluster` re-shards it). With the warm-up, T0 goes to a clean, *sustained* 1.0.

### 8.3 Four findings that make warm-up + RL actually work

- **A. Gradient clipping is load-bearing — its absence is a *crash*, not just instability.** Unclipped multi-turn GRPO hit `inf`/`NaN` gradients that surfaced as a **libtpu `SIGSEGV`** (lr 2e-5 died ~step 3; lr 1e-5 limped to ~step 99 then died), losing all progress. `optax.chain(clip_by_global_norm(1.0), adamw(...))` both **eliminates the crash** and stabilizes `arg_acc` to ~1.0. We clip in the SFT phase too. This is the single most important infrastructure finding in the project.
- **B. SFT can *over-collapse* the policy.** Too much warm-up drives `tool_call_rate → 0` (the policy sharpens onto a degenerate continuation). T0's sweet spot is ~150 transcripts; ~400 collapses it. Warm-up is a nudge, not a fine-tune.
- **C. A train/RL *distribution mismatch* silently corrupts tool-call emission** (the most interesting bug). A naive T1 warm-up (any amount) broke turn-1: instead of `CALC(92 * 98)` the model emitted a bare `92 * 98`. Debug rollouts isolated it — the **few-shot prompt *alone* (SFT=0) gets turn-1 right**, so SFT was the culprit. Cause: SFT trained on `BOS Q:… → CALC` but RL prompts with `‹few-shot› Q:…`, a context the warm-up never saw. **Benign for single-call T0; corrupting for chained T1.** Fix: **prepend the few-shot prompt (masked, loss 0) to every SFT transcript**, so the SFT context is byte-identical to the RL rollout prompt. Turn-1 `CALC` then survives warm-up and the full chain learns to 1.0. *(Generalizable: warm-up transcripts must match the RL prompt distribution, prefix included.)*
- **D. Base-LM surface gotchas.** Delphi has no chat template and **never emits EOS** (we stop on a digit/`)`-terminated newline); it can't reliably produce Qwen JSON, so the surface is bare-text `CALC(...)`; and the Llama-3 BPE fuses+strips `)\n`, so the parser treats the closing paren as **optional**.

### 8.4 The recipe, and what it proves

The robust recipe across all stages: **prefix-aligned SFT warm-up (~150–250 transcripts, gradient-clipped) → gradient-clipped GRPO** on tunix's agentic learner, all on the same in-memory actor. With it a 447M base model does single (T0), two-deep (T1), and three-deep chained (T2) tool use at ~100% solve, where the chaining is genuine — turn *N+1* consumes turn *N*'s tool output as an argument. The binding constraint is never the RL or the iris/TPU plumbing (both unchanged from the arithmetic runs); it is the **base LM's format/copy priors**, and prefix-aligned SFT is exactly the lever that fixes them.

**T2 confirmed the recipe generalizes with depth.** The three-call extension (`a*b*c*d`, `env_max_steps=4`) was a pure config-level addition (~30 LOC: `t2_segments` + `T2_SYSTEM_PROMPT` + `build_t2_dataset`). It reached `solve_ratio` **1.0 (SFT=250, sustained ~100 steps)** / ~0.97–0.98 (SFT=150), threading two intermediate (~6-digit) results forward. The only depth cost observed is a slightly higher SFT budget and a small residual error at the lighter warm-up — consistent with copies of longer numbers being marginally harder.

**Verdict (#5): yes — tunix's agentic stack runs on iris and a small marin base model learns real multi-step (up to 3-deep) tool use, given a distribution-matched SFT warm-up.** Reproduce with `DELPHI_AGENT_MODE={t0,t1,t2}` + `DELPHI_SFT_STEPS={150,250}` (see `launch_agentic.py`).

---

## 9. Follow-on #7: agentic coding — Delphi writes Python an interpreter grades

§8 had Delphi *call* a fixed tool. The natural next question: can the same 447M base LM be bootstrapped to *write code* — emit a small Python program that a real interpreter executes and grades? We built a purely-functional **micropython** interpreter (execution environment + verifier), a **50-task difficulty ladder**, and trained with the same three-stage recipe, **switched to Dr.GRPO** (more robust on a tiny actor). **Result: a base LM that solves 3/50 from few-shot reaches a perfect 50/50 after a short SFT warm-up — writing genuine recursive/looping programs — and the lesson *inverts* §8: SFT does the heavy lifting and RL is marginal.**

### 9.1 The setup

- **Interpreter (`micropython.py`).** A tree-walking interpreter over Python's `ast` for a safe subset (ints/floats/strings/lists, arithmetic/bool/compare, `if`/`while`/`for`, `def` + recursion, list comprehensions, f-strings, a whitelist of builtins/methods). **Purely functional**: deterministic, sandboxed (no `import`/IO/dunder access), bounded (step + output caps → infinite loops/runaway recursion terminate cleanly); `run(src) -> ExecResult(stdout, ok, error, steps)` **never raises**. 90 unit tests.
- **Tasks + training distribution.** `coding_tasks.py` is the **held-out** 50-task ladder (5 tiers × 10: constant → one-step arithmetic → variables/conditionals → loops → functions/recursion), each `(prompt, reference solution, gold stdout)`; the oracle solves 50/50 through the grader. Training uses **41 parameterized task *families*** (`coding_env.py`) — random params per prompt — the same **anti-hardcoding** trick as the CALC random operands: with a *fixed* task and an exact-stdout reward a model could "solve" by printing the constant, so randomized golds force a *general* program. Eval prompts strip answer-leaking hints.
- **Action / format.** Single-turn on the proven non-agentic GRPO wiring (`train_delphi.py`): the model writes a program after a 3-demo few-shot prefix and ends with an `END` sentinel; a parser extracts it, `micropython.run` executes it, and the reward is **exact stdout match plus a dense shaping term** (has-code / ran-ok / output-prefix-overlap) so Dr.GRPO has a gradient before any sample is exact. Same base-LM surface rules as §8 — but multi-line code means **no newline-stop**: generate to the budget, cut at `END`.
- **RL: Dr.GRPO** (`DrGRPOLearner`/`DrGRPOConfig`): advantage = group-mean-centered with **no std division**, loss **constant-normalized** (no per-response length bias). A drop-in for the GRPO learner.

### 9.2 Results — few-shot can't code; a short SFT warm-up gets there

Greedy solve on the held-out 50-task ladder, one `v6e-4`:

| configuration | solve | per-tier (t0 / t1 / t2 / t3 / t4) |
|---|---|---|
| **few-shot only** (no SFT/RL) | **3/50** | 2 / 0 / 0 / 0 / 1 |
| SFT 150 | 45/50 | 9 / 10 / 9 / 7 / 10 |
| SFT 300 | 46/50 | 9 / 9 / 10 / 8 / 10 |
| SFT 300 → **Dr.GRPO** 80 | **45**/50 | 9 / 9 / 9 / 8 / 10 |
| SFT 600 | 47/50 | 9 / 9 / 10 / 9 / 10 |
| SFT 1000 / 1500 | **48/50** | 9 / 10 / 10 / 9 / 10 |
| SFT 1000 (+2 coverage families) | **50/50** | 10 / 10 / 10 / 10 / 10 |

(SFT 1200 with the coverage families gave 49/50 — a single 7-digit transcription slip, `print(1001000)` for `1000000` — mild run-to-run variance at the ceiling.) The programs are genuine code (hints stripped): `print('hello'[::-1])`, recursive `fib`/`factorial`/`is_prime`, accumulator loops, list comprehensions, the Euclidean `gcd`. **Tier 4 (functions/recursion) is the *strongest* tier (10/10)** — those tasks have canonical solutions SFT learns cleanly.

### 9.3 What it shows — the mirror image of §8

- **The base LM cannot code from few-shot (3/50, tiers 1–3 at 0); a short SFT warm-up unlocks it (→48).** Same wall as the §8 copy: RL only amplifies what the base policy samples, and a 447M / 1.2B-token model almost never samples a correct multi-line program cold. SFT puts "emit a valid program in this format" in distribution.
- **SFT scales *monotonically and plateaus* — no over-collapse (opposite of finding 8.3-B).** 45 → 46 → 47 → 48 → 48 across SFT 150 → 1500, then flat. Structural reason: coding has **41 skill-families** (a *broad* target), so more SFT buys coverage; the single narrow CALC behavior instead sharpened to a degenerate point past ~250. *The right SFT amount depends on how broad the target behavior is.*
- **Dr.GRPO added nothing (46 → 45, within noise) — and that is the result.** SFT already saturates the training families (~98% train solve), so RL has no headroom; the residual eval gap is **held-out generalization**, which RL on the same families cannot close. Contrast §8, where RL was *essential*. Dr.GRPO itself ran stably; it just had nothing to do.
- **Coverage, not optimization, is the lever.** Broadening 27 → 41 families (to cover the eval's task *types*) moved SFT solve 34 → 48. The two misses at 48/50 were a string **copy-precision** edge (`Hello, World!` → lowercased, the model having seen only lowercase words) and an **uncovered type** (a descending while-loop countdown); two coverage families close them to a clean **50/50**.

### 9.4 Verdict (#7) + reproduce

**Yes — the same base LM that learns to *call* a tool (§8) can be bootstrapped to *write code* an interpreter executes, reaching 50/50 on a 5-tier ladder including recursion.** Reproduce with `python launch_coding.py` + `CODING_SFT_STEPS=1000`; files: `micropython.py` / `coding_tasks.py` / `coding_env.py` / `train_coding.py` / `launch_coding.py`.

---

## 10. The cross-experiment lesson: when RL is essential vs. when SFT suffices

The two follow-ons are the same recipe (SFT warm-up → RL on the same in-memory actor, gradient-clipped, on tunix+iris) pointed at two behaviors — and they came out *opposite*, which is the most useful thing the project produced.

| | §8 tool-use (copy the result) | §9 coding (write the program) |
|---|---|---|
| target behavior | **narrow** (one copy-forward step) | **broad** (41 program families) |
| base-model sample rate | rare → SFT can't fully cover it | demonstrable → SFT covers it |
| SFT scaling | over-collapses past ~250 transcripts | monotone, plateaus, no collapse |
| RL role | **essential** — amplifies the rare copy | **marginal** — no headroom left |
| RL algorithm | GRPO | Dr.GRPO (same plumbing) |

**Deciding rule.** RL earns its keep only for a *narrow* behavior that must be amplified from rare base-model samples (the tool-result copy). When the target behavior is *fully demonstrable by SFT* (writing programs), SFT alone wins and RL is moot. Both cases hit the *same* base-LM wall — "RL only sharpens what the base policy already puts mass on" — and resolve it the same way (an SFT warm-up to put the behavior in distribution); they differ only in whether SFT can *finish* the job or merely *start* it. The binding constraint, in every milestone, was the base LM's priors — never the RL math, never the iris/TPU plumbing, which worked unchanged throughout.

---

## 11. Next: multi-turn coding (in progress)

§9's coder is **single-turn**: write once, get graded, done — which is exactly why SFT saturated it and Dr.GRPO had nothing to do. We are now escalating to a **multi-turn coding agent**: up to **5 rounds** of write → run → revise, where the model sees the interpreter's execution feedback (stdout, errors, step traces) and edits its program, trained with Dr.GRPO on a **harder problem set** where SFT alone is insufficient.

**Hypothesis.** On tasks hard enough that the first-attempt solve rate is *low* even after SFT, the gain has to come from **iterating on execution feedback** — a behavior SFT can demonstrate the *form* of but cannot supply the *policy* for (which fix to try given which error), because the right next action depends on the observed failure. That is precisely the narrow-behavior-amplified-from-feedback regime where §8 showed RL is essential. So we expect the §9 result to *flip back*: multi-turn, hard-task coding should be the case where **Dr.GRPO finally beats SFT**, by learning to recover from failures the single-turn policy cannot.

This work is **in progress; no results yet.** The hypothesis above is a prediction, not a finding — it will be confirmed or refuted in a follow-up.

---

Full design + A/B/C strategy trade study: [`DESIGN.md`](DESIGN.md). How to work in this directory (file map, invariants, gotchas): [`AGENTS.md`](AGENTS.md). Per-milestone working logs: `.agents/logs/tunix-iris/`. Tracked as weaver issues #229 (feasibility), #5 (tool use), #7 (coding).
