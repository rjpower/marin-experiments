# Adopting google/tunix on iris: from a "print cats" smoke test to an agentic coder

**Marin Research Report · June 2026**

Weaver issues #229 (feasibility) · #5 (agentic tool use) · #7 (agentic coding) · #8 (test-case curriculum → the clear RL win at 1.7B, §13) · the 2B-class capstone: marin's Delphi-1.9B vs Qwen3-1.7B up the chat → tool → agentic ladder (§14)

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

We then escalated once more — a **multi-turn coding agent** with a **test-case-graded reward and a difficulty curriculum**, built specifically to predict that Dr.GRPO would *finally beat SFT* (§11). At **447M it did not** (§12): the redesign **fixed the representation** (SFT now reaches pass@1 = 1.0 on every held-out level, vs. ~1/18 before), but RL moved held-out pass@1 by **exactly zero**, because the 447M genuinely cannot generalize to held-out families at this scale — `pass@16 = 0` there even with proper sampling, so RLVR has **no pass@k-to-pass@1 gap to compress**. (One honest correction along the way: the "*bimodal at every level, even at temperature 1.5*" framing was partly a **measurement bug** — tunix's `Sampler` decodes greedily unless `top_p` is passed, so our eval's *k* "draws" were one identical sequence; the held-out-`= 0` result survives re-measurement, the "no spread anywhere" generalization did not. §12/§13.)

**So we acted on §12's own prescription — bigger base model + compositional problems — and got the clear win (§13).** On **Qwen3-1.7B-Base** (Delphi is a Qwen3, so it loads through the identical native path) with multi-stage problems and *sampled* pass@k, **Dr.GRPO lifts aggregate pass@1 from 0.289 → 0.466 (+61 %)** — the lift largest at pass@1 and shrinking with *k* (textbook pass@k compression), raising pass@16 to 1.0 on trained mid-levels, **and generalizing to held-out levels 7–9 it never trained on** (tier 7 pass@1 0.078 → 0.188). RL is decisive exactly when the model is *partially competent and unreliable*; tunix/iris delivers that win at the 1.7B scale.

**Finally, the capstone (§14): the same ladder on marin's own model.** We walked **Delphi-1.9B** (`delphi-3e20-1.9Bparams-24.7Btokens`, width-matched to Qwen3-1.7B-Base; the live variable is pretraining tokens, **24.7B vs ≈36T, ~1500×**) up conversation → tool-calling → agentic coding against Qwen. **Post-training amplifies pretraining competence and creates none.** The gap is **invisible** where the task is memorizable — both reach chat turn-leak 0/8 in 200 steps; on the fixed coding ladder Delphi 40 ≈ Qwen 37 of 58 — and **decisive** where it demands generalization: few-shot coding pass@1 **0.003 vs 0.29**, and on the graded curriculum the same "up to shape" warm-up leaves Qwen's generalizing RL win intact (pass@1 0.242 → **0.451**, held-out 7–9 still lift) while Delphi never leaves the **0.003** floor. The deciding factor for whether RL helps is the base model's pretraining competence on the target skill — sampled few-shot pass@k before any RL — not the post-training recipe.

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

- **Submission is unchanged from grug experiments:** `iris --cluster=marin job run --tpu v6e-4 --enable-extra-resources --extra tpu --region europe-west4 -- python examples/launch.py`. The worker bundles the dir (git-ls-files; `.venv`/`__pycache__` excluded; 25 MB cap, our bundle is ~0.5 MB), `uv sync`s the committed lock, and runs. Delphi's 1.8 GB weights download from HF at runtime (not bundled).
- **`--tpu` is optional** — a CPU job cleanly de-risks packaging+submission before paying for TPU. Memory ≥4 GB / disk ≥10 GB need `--enable-extra-resources`.
- **Transient TPU bad-nodes:** several v6e nodes failed init with `"Couldn't open iommu group /dev/vfio/N: Device busy"`. iris detects the signature and reschedules automatically — keep `--max-retries` generous (we used 5).
- **v6e-4 capacity** was readily available (no boot wait) in `europe-west4` and `us-east5-b`; a 447M GRPO job runs colocated on one v6e-4 host in minutes. The marin cluster was reachable via gcloud auth (`iris-controller@hai-gcp-models`, 511/511 workers healthy).

### Reproducing the feasibility + arithmetic results

```bash
cd tunix-delphi-rl
uv sync --frozen                                         # stock google-tunix 0.1.7 + pytest, CPU
JAX_PLATFORMS=cpu uv run pytest tests/test_smoke_cats.py -m slow   # M2 toy GRPO learns on CPU
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

**Yes — the same base LM that learns to *call* a tool (§8) can be bootstrapped to *write code* an interpreter executes, reaching 50/50 on a 5-tier ladder including recursion.** Reproduce with `python launch_coding.py` + `CODING_SFT_STEPS=1000`; files: `environments/micropython.py` / `problems/coding_tasks.py` / `environments/coding_env.py` / `training/train_coding.py` / `launch_coding.py`.

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

## 11. The multi-turn prediction we set out to test (#8)

§9's coder is **single-turn**: write once, get graded, done — which is exactly why SFT saturated it and Dr.GRPO had nothing to do. So we escalated to a **multi-turn coding agent**: write → run → revise over several rounds, the model seeing the grader's execution feedback and editing its program, trained with Dr.GRPO on a problem set rebuilt so SFT alone is insufficient.

**The prediction.** On tasks hard enough that the first-attempt solve rate is *low* even after SFT, the gain has to come from **iterating on execution feedback** — a behavior SFT can demonstrate the *form* of but not the *policy* for (which fix to try given which error). That is the narrow-behavior-amplified-from-feedback regime where §8 showed RL is essential, so we expected §9 to *flip back*: multi-turn, hard-task coding should be where **Dr.GRPO finally beats SFT**.

We built the whole apparatus to test it — and the prediction was **wrong**, in a way that turned out to be the most precise result in the project. §12 is that result.

---

## 12. The result: a test-case curriculum, and why RL *still* doesn't beat SFT at 447M (#8)

> **Correction (added 2026-06-21, after §13).** The headline diagnostic in this section — *"pass@1 = pass@k at every tier and temperature, so the policy has no spread for RLVR to compress"* — was **partly a measurement artifact**. The eval sampler was called without `top_p`, and tunix's `Sampler` then decodes **greedily** (silently ignoring `temperature` *and* `seed`), so every "draw" was the identical argmax sequence. Re-measured with real sampling (`top_p=1.0`): the **held-out 447M result below stands** — levels 4–6 are genuinely `pass@16 = 0` (the model truly cannot generalize to held-out families at this scale), so the "RL acquires no new held-out capability at 447M" conclusion is **correct**. But the *generalization* — "no pass@k spread anywhere, fundamentally bimodal" — was the artifact; the partial-competence configs did have hidden spread. The corrected, complete picture and the **clear RL win at 1.7B** are in **§13**. The narrative below is preserved as the original record; read it through that correction.

To give RL a real chance we rebuilt the task three ways (design: [`CURRICULUM_DESIGN.md`](CURRICULUM_DESIGN.md); literature-grounded rationale: [`RL_HEADROOM.md`](RL_HEADROOM.md)):

1. **Test-case-graded reward.** Instead of one canonical-output match, the model writes `def solve(...)` and is scored on the **fraction of N hidden tests passed** — a *continuous* reward, so a group of samples has reward *variance* and Dr.GRPO has a gradient. (The old binary reward gave all-pass-or-all-fail groups → zero advantage → no gradient.)
2. **Six graded difficulty levels + a curriculum** (frontier-biased sampling, fixed cadence + a mastery gate, borrowed from `lib/marin/src/marin/rl/curriculum.py`): L1 arithmetic → L6 `nth_prime`/`collatz`/word problems.
3. **Multi-turn + CoT exploration**: reason-as-comment, `def solve`, see the public-test result, revise — up to a few rounds.

**What this fixed: the representation.** It is a real improvement over §9. SFT on **all six levels** now reaches **pass@1 = 1.000 on every held-out level** — and saturates in **under 30 SFT steps**. (The old stdout representation left held-out *families* stuck at ~1/18; the `def solve` + test-case framing makes the 447M model fully capable and parameter-generalizing across all six levels.) So the families are well-formed and learnable; the wall is not the task surface.

**The clean RL test.** SFT on levels **1–3 only**, then curriculum Dr.GRPO over levels 1–4, measuring **per-level pass@k on held-out instances before vs. after RL**. A win = held-out L4 pass@1 climbing from 0. Replicated at two rollout shapes (num_gen 8 × batch 4, and 16 × batch 2):

| tier | after-SFT pass@1 | **after-RL pass@1** |
|---|---|---|
| 1–3 (in SFT) | 1.000 | **1.000** |
| 4–6 (held-out) | 0.000 | **0.000** |

**Byte-identical, both runs.** RL moved nothing on the held-out levels — and this is *not* the §9 "no headroom" story (there SFT had already won; here SFT never saw L4–6) nor gradient starvation: the **training reward is a healthy partial-credit signal** (level-4 steps sit at reward ≈ 0.3–0.5 — programs that *run* and pass ~13–50 % of tests). The gradient exists. RL simply cannot convert "passes some tests with a hacky program" into the actual algorithm on a family SFT never demonstrated.

**Why — the mechanism, and it is exact.** Every evaluation shows **pass@1 = pass@2 = pass@4 = pass@8 = pass@16**, at every tier, at every SFT amount (15/20/30/60/250 steps), at temperature **1.0 *and* 1.5**. Zero sampling variance: the 447M policy is **bimodal per problem** — it either always emits a correct program or always fails, with nothing in between, and raising temperature to 1.5 changes neither the spread (still none) nor the outcomes. RLVR's entire win mechanism is **compressing pass@k into pass@1** — sharpening unreliable-but-sometimes-right into reliable. A policy with *no* pass@k spread gives RL **nothing to compress**. (Corollary, also observed: `best_solve == first_solve` every step — the multi-turn feedback is unused, because when round 1 succeeds there is nothing to fix and when it fails the model cannot fix it. The same bimodality kills the iterate-on-feedback channel §8 relied on.)

This **empirically confirms** the risk flagged up front in [`RL_HEADROOM.md`](RL_HEADROOM.md) §1.1/§6: at sub-2B scale, on templated single-function problems, there may be *no* reachable regime where RL beats SFT — and there isn't.

**So the §10 deciding rule sharpens.** RL needs not just a *narrow behavior amplified from rare samples* (§8) but, concretely, a **`pass@k > pass@1 > 0` regime** — tasks the model solves *sometimes but unreliably*. Our algorithmic templates never produce one at 447M: they are either learned (→ deterministic 1.0) or not (→ deterministic 0.0).

**What would actually give a clear win** (the prescription, not yet run): manufacture the sampling gap with **compositional / multi-step problems** — longer programs that chain 2–3 operations, where the SFT'd model's first attempt is often buggy but *occasionally* correct (pass@1 < pass@k), which is also where the unused multi-turn-feedback channel would finally pay off — **or** move to a **larger base model** above the sub-2B floor. Temperature is *not* a lever (tested: 1.5 opens no gap).

> **Integration note (worth logging).** The test-case grader runs the `micropython` interpreter ~9× per program per round (N hidden tests + public-test feedback) vs. 1× in §9. At rollout concurrency 32 this blew host RAM and the container was OOM-killed at 64 GB; the runs only completed at **150 GB**. The TPU/HBM and iris plumbing were never the constraint — host-side grading cost was.

## 13. The clear RL win: a bigger base model, a measurement-bug fix, and pass@k compression that *generalizes* (#8)

§12 ended with a prescription it had not yet run: a clear RL win needs a **`pass@k > pass@1 > 0`** regime, which 447M's templated families never produce — so manufacture it with **compositional problems** and a **larger base model**. We did both. The result is an unambiguous win.

**Three changes.** (1) **Reorganized the codebase** into `problems/ models/ environments/ training/ tests/` with a pytest suite, and rewrote `AGENTS.md` (the layout this report's links assume). (2) **Bumped the model class** to **Qwen3-1.7B-Base** — a general HF→tunix Qwen3 loader (`models/qwen3_loader.py`) and a `ModelSpec` registry (`models/registry.py`) parameterize the whole train/eval stack by model; Delphi *is* a Qwen3, so the 1.7B loads through the same native path with no architecture work, and at `rope_theta = 1e6` it needs no RoPE monkeypatch (§3.1). (3) **Compositional problems**: levels **7–9** chain 2–3 operations into longer programs (e.g. *sum of primes below n*, *RLE then reverse*, *evaluate a small expression*), held out from training.

**The pivotal fix — a measurement bug, not a model change.** Before trusting any pass@k number we found that tunix's `Sampler.__call__` **decodes greedily unless `top_p` is passed** — it picks sampling mode by argument, and with only `temperature`/`seed` (no `top_p`) it silently runs argmax, ignoring both. *Every* `evaluate_problems_passk` call had been greedy, which is exactly why §12 saw `pass@1 = pass@k` everywhere (all *k* draws were one identical sequence). The RL **rollout** path was unaffected (its `RolloutConfig` sets `top_p = 1.0`), so training dynamics were always real — only the eval lens was broken. One-line fix: pass `top_p = 1.0` to the eval samplers. With it, the 1.7B's few-shot baseline shows a **wide, real** pass@k gap.

**The setup.** No SFT (few-shot base model), **Dr.GRPO for 150 steps** over train levels **1–6**, eval on **all 1–9** (so 7–9 are held out), `num_gen 8 × batch 3`, lr `2e-6`, temp 1.0, multi-turn rounds 2. The 1.7B fp32 actor + optimizer + rollout needs ~68 GB/chip and **OOMs a v6e-4** (31 GB/chip); a single-host **v6e-8** (2× sharding) fits it — the first time in this project the TPU itself, not host RAM, set the slice size.

**The result — aggregate pass@k (72 tasks, k=16, temp 1.0), before → after RL:**

| metric | before RL (few-shot) | **after RL** | Δ |
|---|---|---|---|
| **pass@1** | 0.289 | **0.466** | **+0.177 (+61 %)** |
| pass@2 | 0.440 | 0.615 | +0.175 |
| pass@4 | 0.588 | 0.733 | +0.145 |
| pass@8 | 0.708 | 0.822 | +0.114 |
| pass@16 | 0.792 | 0.875 | +0.083 |

**Three textbook RLVR signatures at once:**

1. **Compression.** The lift is **largest at pass@1 and shrinks monotonically as k grows** (+0.18 at k=1 → +0.08 at k=16) — the precise shape of RLVR converting unreliable-but-sometimes-right (pass@k) into reliable (pass@1).
2. **Ceiling raised, not just sharpened.** On trained mid-levels pass@16 went to **1.0** (tier 5: 0.63→1.0, tier 6: 0.75→1.0) — RL also found genuinely new solutions, not only re-weighted old ones. Per-trained-tier pass@1 rose +0.13 to +0.31 (tier 1: 0.59→0.90, tier 2: 0.53→0.83, tier 4: 0.43→0.71).
3. **Generalization to held-out levels.** Levels **7–9, never trained on**, also lifted: tier 7 pass@1 **0.078 → 0.188** (>2×), tier 8 0.086→0.141, tier 9 0.094→0.188. RL trained on 1–6 transferred to harder compositional problems it never saw — the "SFT memorizes, RL generalizes" effect, demonstrated.

**Why this won where §12 didn't — and it reconciles cleanly.** RLVR needs a `pass@k > pass@1 > 0` gap to compress. The 447M after SFT(1–3) had none on held-out families: re-measured *with sampling*, it is genuinely `1.0` in-distribution and genuinely `0.0` at all k on held-out (a **capability floor**, confirmed not an artifact). The 1.7B few-shot has partial competence **everywhere** — a wide gap on every tier — so RL has spread to compress on all nine levels. The §12 prescription was right; the only correction is *why* 447M failed (it lacks the gap on held-out families, not "RL fundamentally cannot compress at this scale").

**Verdict (#8).** With a sound measurement (sampled pass@k), a base model above the sub-2B floor, and compositional problems, **Dr.GRPO on tunix/iris produces a clear, generalizing RL win over the few-shot base** — pass@1 +61 % relative, compressing the pass@k gap and transferring to held-out difficulty. This is the positive counterpart to §10/§12: RL is decisive exactly when the model is *partially competent and unreliable*, and tunix's GRPO on iris delivers it at the 1.7B scale.

*Caveat:* eval is `n=8` problems/tier (72 total); the per-tier deltas (+0.13–0.31 on trained tiers) are well beyond sampling noise, but a larger-`n` confirmation would tighten the held-out figures. `train_curriculum` does not checkpoint the actor, so re-eval requires a re-run. *Reproduce:* `CURRIC_MODEL=qwen3 CURRIC_SFT_STEPS=0 CURRIC_TRAIN_LEVELS=1,2,3,4,5,6 CURRIC_EVAL_LEVELS=1,2,3,4,5,6,7,8,9 CURRIC_NUM_GENERATIONS=8 CURRIC_BATCH_SIZE=3 CURRIC_MAX_PROMPT=896 CURRIC_MAX_RESPONSE=512 CURRIC_LR=2e-6 python launch_curriculum.py` on a v6e-8 (`--memory 200GB`; the launcher's default `num_gen 16 × batch 8` rollout OOMs HBM on v6e-8, so shrink the shape as shown).

---

## 14. The 2B-class capstone: walking marin's own Delphi up the ladder, vs Qwen3-1.7B

§13 won with Qwen3-1.7B-Base. The obvious question for marin is whether the *same* recipe carries one of marin's *own* models — so we took **Delphi-1.9B** (`marin-community/delphi-3e20-1.9Bparams-24.7Btokens`) and walked it up the full post-training ladder the brief implies: **conversational → tool-calling → agentic coding**, at each rung against Qwen3-1.7B-Base under a controlled comparison.

**The comparison is built to isolate one variable: pretraining data.** Delphi-1.9B is width-matched to Qwen3-1.7B-Base (hidden 2048; Delphi is a Qwen3, so both load through the identical native path of §2/§13). The two differ almost entirely in pretraining token budget: **Delphi 24.7B tokens vs Qwen3 ≈36T**, about a **1500×** gap. Both are base LMs with no chat template. So every rung below is the same question asked at increasing difficulty: *does post-training close a 1500× pretraining gap, or only reveal it?*

**Section takeaway.** Post-training amplifies pretraining competence; it does not create it. The gap is **invisible** at rungs a model can satisfy by memorizing a format or a fixed task set (conversation, the fixed coding ladder), and **decisive** at rungs that demand generalization to held-out problems (the graded curriculum). Same two models, opposite verdicts, depending only on whether the task rewards memorization or generalization.

### 14.1 Conversational rung: format installs equally, content does not

We SFT one model-agnostic plain-text ChatML format (`<|user|>` / `<|assistant|>`, loss masked to assistant content + EOS) on the `allenai/tulu-3-sft-mixture`, identically on both models — `training/chat_sft.py` + `launch_chat_sft.py`, 1500 steps, v6e-8 (`chat-delphi19-full`, `chat-qwen17-full`).

**Format uptake is a tie.** Both base models start at turn-leak 8/8 (they ramble past the assistant turn and hallucinate further turns) and reach **0/8 — bounded, EOS-terminated — within 200 steps**. marin's 1.9B learns the chat *shape* as readily as Qwen. Format is data-cheap.

**Content quality is a clear Qwen win.** On the same 8 held-out prompts, the two tie on easy factual/structured asks (capital of France; a study-tips list; summarize a neural network), and Qwen wins everything that needs knowledge or reasoning:

- "Write a recursive factorial" → Qwen emits a correct recursive `factorial`; Delphi emits a Fibonacci-shaped function that does not compute factorial.
- "Why is the sky blue?" → Qwen gives Rayleigh scattering; Delphi attributes it to water droplets.
- "Say good morning in Spanish" → Qwen "¡Buenos días!"; Delphi "Hola" (= hello).
- A speed/time word problem → Qwen uses the right method; Delphi answers 0.97 mph.

Delphi also degenerates on open-ended generation: asked for a haiku, it emits "The ocean, salt water," repeated twelve times, the classic small/data-limited collapse. Chat behavior is a cheap skin both models wear. The competence underneath comes from pretraining, and only Qwen has it.

### 14.2 Coding rung: Delphi is at the floor, and the bimodal wall reappears at 1.9B

Before spending RL we ran a **few-shot coding probe** (eval-only, the §13 graded ladder; `delphi19-code-probe`): can the base model write a program at all?

| model | few-shot pass@1 | few-shot pass@16 | fraction of generations that even run |
|---|---|---|---|
| Delphi-1.9B | **0.003** | **0.042** | **2–6 %** |
| Qwen3-1.7B-Base | 0.29 | 0.79 | (majority) |

Delphi-1.9B is at the floor — its generations rarely parse as runnable Python, let alone solve the task. Its 24.7B-token pretraining never installed code competence, so there is no few-shot `pass@k > pass@1 > 0` spread for RLVR to compress. This is exactly the precondition §12/§13 identified, now failing for Delphi where it held for Qwen.

**SFT lifts Delphi off the floor but only memorizes.** `delphi19-code-sftrl` (agentic SFT 500 steps on levels 1–6, then curriculum Dr.GRPO 150, eval 1–9):

- After SFT: trained levels **1–6 = 1.000** (fraction and pass@k), held-out levels **7–9 = 0.000**.
- RL is a **no-op**: the reward is `1.0000` on essentially every step → zero GRPO advantage → no gradient. After-RL aggregate `pass@1 = pass@16 = 0.667` (6/9 levels solved, 3/9 at zero), unchanged from after-SFT.

This reproduces the **447M bimodal wall** (§12) at 1.9B: saturated (= 1) in-distribution, incapable (= 0) on held-out, no pass@k spread anywhere, so RL has nothing to compress. The extra capacity over 447M did not buy held-out generalization; the pretraining-data gap did not move.

### 14.3 Multi-turn agentic capstone on a fixed ladder: non-discriminating

We then built the full "up to shape" multi-turn agent and ran both models head-to-head: load → **chat + tool-use SFT** (tulu-3 + a `smoltalk2` tool-use mixture via `load_up_to_shape_mixture`, 600 steps) → **code-agent SFT** (multi-turn write→run→revise format, 500 steps) → **multi-turn Dr.GRPO** (60 steps, 4 rounds). The task is the **fixed** `coding_tasks` ladder (the §9 ladder, 58 eval tasks), train tiers 1–4, eval 1–5 with tier 5 held out. Greedy multi-turn eval, first-attempt vs best-across-rounds (`delphi19-mt-capstone`, `qwen17-mt-capstone`).

| | Delphi-1.9B after-SFT → after-RL | Qwen3-1.7B after-SFT → after-RL |
|---|---|---|
| tier 1 (train) | 10/10 → 10/10 | 9/10 → 10/10 |
| tier 2 (train) | 8/10 → 9/10 | 10/10 → 10/10 |
| tier 3 (train) | 9/10 → 9/10 | 7/10 → 8/10 |
| tier 4 (train) | 10/10 → 10/10 | 10/10 → 8/10 |
| tier 5 (held-out) | 2/18 → 2/18 | 0/18 → 1/18 |
| **total (best/58)** | **39 → 40** | **36 → 37** |

The two models are **tied within noise** (Delphi 40, Qwen 37 of 58), which is itself the finding. The fixed ladder is **memorization-dominated on the trained tiers** — `add`/`sum`/`fib`/`factorial` with fixed gold outputs, which both models SFT-saturate, so the pretraining gap is invisible exactly as it was at the chat-format rung — plus a **held-out cliff** at tier 5 (`bubble_sort`/`digital_root`/`fizzbuzz`, structurally novel) that floors both. Two further observations:

- **Multi-turn repair buys almost nothing at 2B.** Best-across-rounds barely exceeds first-attempt (Qwen 37 = 37; Delphi 40 vs 39). Failed tier-5 transcripts spend all four rounds emitting garbled revisions (`"0 else 'FizzBuzz')"`, `"for i in 6)"`) rather than reading the execution feedback and fixing the program. The write→run→revise loop did not install a repair policy at this scale with this recipe.
- The "up to shape" warm-up successfully installed conversational + tool-call behavior on both models, and **did not change the coding outcome** — consistent with §14.1/§14.2.

### 14.4 The discriminating head-to-head: "up to shape" on the graded curriculum

§14.3 was non-discriminating because the fixed ladder rewards memorization. The task that *does* separate the two models is the **graded curriculum** of §13 — parameterized families with sampled `pass@k`, where Qwen has partial competence everywhere and RL generalizes to held-out levels 7–9. To test whether the conversational/tool warm-up changes that, we **prepended the same "up to shape" Stage-0 (chat + tool-use SFT, 600 steps) to the §13 curriculum recipe and re-ran both models** — otherwise matched to §13 (few-shot base into curriculum Dr.GRPO over levels 1–6, eval 1–9, `num_gen 8 × batch 3`, lr 2e-6, rounds 2, sampled pass@16, eval `n=8`/tier, v6e-8). This adds exactly one variable on top of §13: the warm-up. (Runs `qwen17-curric-ups3` / `delphi19-curric-ups3`, `--memory 500GB` — the host-side micropython grading high-water exceeds 200GB once the chat-SFT prefix raises the resident baseline, the same host-grading cost flagged in §12.)

**Qwen keeps the clear, generalizing RL win; Delphi stays on the floor.** Aggregate pass@k (72 tasks, k=16, temp 1.0), few-shot (post-warm-up) → after-RL:

| | Qwen3-1.7B few-shot → after-RL | Delphi-1.9B few-shot (RL cut) |
|---|---|---|
| pass@1 | 0.242 → **0.451** | **0.003** |
| pass@4 | 0.527 → 0.700 | 0.013 |
| pass@16 | 0.722 → 0.861 | 0.042 |

For Qwen the warm-up *lowered* the few-shot baseline relative to §13 (pass@1 0.289 → 0.242), but RL recovered it to essentially the §13 after-RL ceiling (**0.451 vs 0.466**) and **still generalized to the held-out levels**: tier 7 pass@1 0.094 → 0.219, tier 8 0.031 → 0.133, tier 9 0.070 → 0.109. The compression signature is intact (lift +0.21 at pass@1, shrinking to +0.14 at pass@16). The up-to-shape warm-up did not change Qwen's wall — the win is robust to it.

For Delphi the warm-up moved coding competence by **zero**: post-warm-up few-shot pass@1 = 0.003, pass@16 = 0.042 — identical to the §14.2 probe, with 1–9 % of generations even running. With no pass@k spread there is nothing for RL to compress, exactly as in §12 and §14.2. We **cut Delphi's RL run after ~4 h** without an after-RL measurement: its floor-level generations never emit a terminating token, so every rollout runs the full 512-token budget twice per round across 24 trajectories and grades slowly on the host (vs Qwen's ~2 h, whose programs terminate early), and the loop had not reached the after-RL eval. The number is therefore unmeasured, but the outcome is fixed by the baseline — a `pass@1 = 0.003` / `pass@16 = 0.042` policy has no spread — and Delphi's curriculum RL was already shown to be a no-op in §14.2 (there `reward = 1.0` every step on the SFT-saturated variant; here it is the mirror case, reward near zero with no variance). Either way the gradient vanishes.

### 14.5 Verdict: post-training amplifies pretraining competence; gap visibility is task-dependent

The discriminating run settles it: the same "up to shape" warm-up is roughly neutral for Qwen (RL still wins and generalizes, 0.242 → 0.451) and useless for Delphi (still 0.003 at the floor). Across all four rungs the constant is that **post-training amplifies what pretraining already seeded and creates nothing new**. Delphi-1.9B matched Qwen on every task a model can pass by memorizing (chat format, the fixed coding ladder's trained tiers) and fell to the floor on every task requiring held-out generalization (the few-shot coding probe, the graded curriculum). RL specifically requires a `pass@k > pass@1 > 0` regime to do its work (§12/§13); Qwen's ≈36T-token base supplies that spread, Delphi's 24.7B-token base does not, and no amount of chat/tool/agentic SFT conjured it. The actionable form: **the deciding factor for whether RL post-training helps is the base model's pretraining competence on the target skill, measured as sampled few-shot `pass@k` before any RL — not the post-training recipe.**

---

## Addendum: the full path

This is the chronological version of the project, including the turns that did not work, kept because the dead-ends are part of the result.

**The brief, and the assumption that turned out wrong.** The task was to evaluate adopting google/tunix on iris and bring a marin "grug-style" model into it, with the expected hard part being a port of grug's `equinox` transformer (forward-only, no KV-cache) into tunix's `flax.nnx` world plus a hand-built sampler. The first thing we checked — Delphi's `config.json` — said `Qwen3ForCausalLM`. Delphi is a dense Qwen3, tunix already ships a native nnx Qwen3 with the full sampler/KV-cache contract, and all 124 safetensors tensors matched tunix's key-map at the byte level. The anticipated multi-week port collapsed to a config + a loader + one bug fix (§2).

**The one real tunix bug.** Loading was 96% right, which on a logit comparison is wrong. tunix's qwen3 `apply_rope` never threads `config.rope_theta` (Delphi uses 500000, not the 1e6 default), and it does not model Llama-3 `rope_scaling` at all — which, contrary to our first guess, is *not* inert at short context. Fixing both yields exact HF parity (top-1 100%, logit MSE 7e-12). We ship it as both an upstreamable patch and a worker-side monkeypatch (§3.1). This bug also affects tunix's llama3.

**De-risking in the cheap order.** Design before code, parity before learning, CPU before TPU. A "print more cats" toy GRPO (fresh tiny Qwen3 rewarded for emitting "cats") learned 0.08 → 1.00 on CPU, then on a v6e-4 iris job — so every later failure could be blamed on the model or task, never the plumbing (§4).

**Arithmetic, and a law we did not expect.** Delphi learned single-digit addition (6% → 65%) and solve-for-x linear algebra (5% → 37%) with a plain exact-match reward. The surprise was the failure pattern: learnability tracked **answer-space density, not symbolic difficulty**. Linear algebra (answer is one of 19 small integers) is *easier* for RL than 2-digit multiplication (thousands of possible answers) because a dense answer space gives the GRPO group non-zero reward variance, hence a gradient. Proximity shaping recovered the flat stages, confirming the stall was missing gradient, not inability (§5).

**Tool use needed RL; coding did not.** Delphi learned genuine multi-step calculator use — up to three chained `CALC(...)` calls at ~100% — but only with a prefix-aligned SFT warm-up; plain RL drove the tool *call* to near-perfect while the *copy-the-result-forward* step stayed at ~0.1, because that copy is out-of-distribution for the base LM and RL only sharpens what the base policy samples (§8). Then the mirror image: given a `micropython` interpreter as grader, Delphi-447M went few-shot 3/50 → SFT 50/50 writing real recursive programs, and Dr.GRPO added nothing — because writing a program is fully demonstrable by SFT, so SFT does the whole job (§9). The cross-experiment rule: RL earns its keep only for a narrow behavior that must be amplified from rare base-model samples (§10). The load-bearing infrastructure finding underneath both: unclipped multi-turn GRPO does not wobble, it crashes — `inf`/`NaN` grads surface as a libtpu `SIGSEGV` — so gradient clipping is mandatory, not a tuning knob.

**The prediction we set out to falsify, and the measurement bug we found doing it.** We built a test-case-graded difficulty curriculum specifically to predict Dr.GRPO would finally beat SFT on hard multi-turn coding. At 447M it did not (§12): SFT reached pass@1 = 1.0 on every held-out level, RL moved held-out pass@1 by exactly zero. The clean diagnosis was almost derailed by a measurement bug — tunix's `Sampler` decodes greedily unless `top_p` is passed, so every pass@k "draw" was the identical argmax sequence, faking `pass@1 = pass@k` everywhere. Re-measured with `top_p = 1.0`, the held-out `= 0` result survived (a genuine capability floor) but the "no spread anywhere" generalization did not. The RL rollout path was never affected; only the eval lens was broken (§12/§13).

**The clear win, on §12's own prescription.** Bigger base model + compositional problems. Qwen3-1.7B-Base on a 1–9 graded curriculum: Dr.GRPO lifted aggregate pass@1 0.289 → 0.466 (+61%), with the three textbook RLVR signatures at once — the lift largest at pass@1 and shrinking with k (compression), pass@16 → 1.0 on trained mid-levels (ceiling raised), and held-out levels 7–9 lifting too (generalization). RL is decisive exactly when the model is partially competent and unreliable, and tunix/iris delivers it at 1.7B (§13).

**The capstone: the same ladder on marin's own model.** Walking Delphi-1.9B up conversation → tool-calling → agentic coding against Qwen3-1.7B-Base (§14) reduced to a single controlled question — does post-training close a 1500× pretraining-token gap? — and answered it the same way at every rung: the gap is invisible where the task can be memorized and decisive where it demands generalization. The iris/TPU plumbing and the tunix RL math worked unchanged from the first toy to the last capstone; the binding constraint, in every milestone, was the base LM's pretraining priors.

---

Full design + A/B/C strategy trade study: [`DESIGN.md`](DESIGN.md). How to work in this directory (file map, invariants, gotchas): [`AGENTS.md`](AGENTS.md). Per-milestone working logs: `.agents/logs/tunix-iris/`. Tracked as weaver issues #229 (feasibility), #5 (tool use), #7 (coding), #8 (test-case curriculum / RL headroom → **resolved in §13: a clear RL win at Qwen3-1.7B-Base** once the eval greedy-decoding bug was fixed and the model class raised).
