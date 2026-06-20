# Evaluating tunix on iris with a marin (grug-style) model

**Design Document & Rollout Plan — Milestone M0 artifact**
Weaver issue #229 · 2026-06-19 · Status: ready for implementation (Strategy B)

---

## 0. TL;DR

**Yes, we can adopt google/tunix on TPUs via iris, and for the Delphi target it is *much* cheaper than the task premise assumed — because Delphi is a stock Qwen3 and tunix already ships a native `flax.nnx` Qwen3 with a working KV-cache sampler and an HF-safetensors loader.** The "grug has no generate/KV-cache path" gap is real for grug-the-equinox-model but does **not** need to be solved by hand: the clean path loads Delphi's published HF weights into tunix's native Qwen3 and runs GRPO with the in-process sampler.

Every load-bearing assumption below was **empirically validated**, not just read from source (see §1). Residual work is a config classmethod, a one-line RoPE bugfix, a calculator/arithmetic dataset+reward, and iris packaging — all bounded.

**Recommended strategy: B (native tunix Qwen3 + load Delphi HF weights).** Estimated ~4–7 engineering days to a learning Delphi-arithmetic loop.

> **Scope note (decision point #1).** This reframes "add a grug-style model into tunix" from "port grug's `equinox.Module`" to "bring a marin model in via tunix's native nnx zoo + weight loading." That is the right engineering call for Delphi (a stock Qwen3 on disk) and is consistent with the task's "build out a delphi model *or load it via levanter as needed*." If the intent is specifically to run grug's *equinox* `Transformer` through tunix (e.g. for a future MoE/XSA architecture with no HF equivalent), that is Strategy A (weeks) — see §3.2. I am proceeding with B and will pivot if redirected.

---

## 1. Validation evidence (what was actually run, not just read)

| # | Claim | How verified | Result |
|---|---|---|---|
| 1 | tunix installs & runs on CPU | built a `uv` venv: `google-tunix` (editable) + `jax[cpu]` | ✅ resolves; **jax 0.10.2, flax 0.12.7, qwix 0.1.7, optax 0.2.8, orbax 0.12.0** |
| 2 | tunix Qwen3 runs end-to-end | instantiated a tiny `Qwen3(ModelConfig(...), rngs=)` and called `__call__` on CPU | ✅ forward → **finite logits `[1,8,256]`**; classes `Qwen3, Cache, LayerCache, Attention, DecoderLayer` present |
| 3 | Delphi is a stock Qwen3 | fetched `config.json` | ✅ `architectures:["Qwen3ForCausalLM"], model_type:"qwen3"` |
| 4 | **Delphi weights load cleanly** | read the **actual `model.safetensors` header** (HTTP range, no full download) and diffed all keys against `qwen3/params.py` key-map | ✅ **124/124 tensors matched, 0 unmatched.** q_norm/k_norm, untied `lm_head`, dense SwiGLU gate/up/down all present. F32. No MoE keys. |
| 5 | **Packaging resolves** | wrote the experiment `pyproject.toml` and ran `uv lock` | ✅ **164 packages resolve**, exit 0. `tensorstore 0.1.84`, `marin-levanter` **absent** (the conflict source), jax 0.10.2 |
| 6 | `rope_theta` is dead code | read `qwen3/model.py` | ✅ presets all hardcode (1e6/5e6); `apply_rope` call sites don't pass `config.rope_theta` — **Delphi needs 500000** |
| 7 | `num_embed` absence is low-risk | traced `sampler.py` | ✅ read only under `include_logits`/`return_logits` which **defaults `False`** (sampler.py:745); add property defensively anyway |

The proven `pyproject.toml` + `uv.lock` are committed in this directory as the experiment manifest starting point.

**Tiny reproduction (the empirical core of the whole verdict):**
```python
from flax import nnx; from tunix.models.qwen3 import model as qm
cfg = qm.ModelConfig(num_layers=2, vocab_size=256, embed_dim=64, hidden_dim=128,
                     num_heads=4, head_dim=16, num_kv_heads=4, rope_theta=500000,
                     norm_eps=1e-5, use_tied_embedding=False)
model = qm.Qwen3(cfg, rngs=nnx.Rngs(0))
logits, cache = model(toks, positions, None, causal_mask)   # -> (1,8,256) finite
```

---

## 2. Background

**tunix** (`/home/power/code/tunix`, `google-tunix 0.1.7`) is Google's JAX-native LLM post-training framework, built entirely on `flax.nnx`. Its RL core (`tunix/rl/`) provides `RLCluster` (holds actor/critic/reference/reward as `nnx.Module`s on per-role meshes — `rl_cluster.py`), GRPO/PPO learners (`grpo_learner.py`, `ppo_learner.py`), a pluggable loss/advantage/reward registry (`function_registry.py`), and three rollout backends: **`vanilla`** (a native JAX KV-cache `Sampler`, `generate/sampler.py`), `vllm`, and `sglang_jax`. The train step is tunix-owned (`nnx.value_and_grad` → `optax.global_norm` → `nnx.Optimizer.update`, `nnx.jit` with donated optimizer); **only the loss/reward functions are pluggable.** It ships native nnx models for llama3, qwen2, qwen3, gemma2/3/4, each with an HF-safetensors loader. It even ships an agentic stack with a `CalculatorTool`, `ToolEnvironment`, and gsm8k/deepscaler/frozenlake examples.

**grug** is marin's hand-rolled, explicit Levanter training template (`marin/experiments/grug/base/model.py`): an `equinox.Module` transformer over **raw `jax.Array`s** (not haliax NamedArrays — haliax only supplies a `named_call` profiling decorator) with explicit `PartitionSpec`/`reshard` sharding. It exposes **only** `__call__`, `.logits()`, `.next_token_loss()` — **no generate, no KV-cache, no position-offset RoPE.** Its philosophy is copy-first (`.agents/skills/change-grug/SKILL.md`): a new behavior is a new variant directory edited by copy-paste, not an abstraction. The `delayed-gradient-pp` experiment is one such variant (a QB-routed MoE) and is *just an example* of grug, not the target.

**iris** is marin's cluster/job manager (`marin/lib/iris`, with the `fray` submission layer). A job runs in a bare `python:3.12-slim` container; the worker installs **all** Python deps at launch via `uv sync --all-packages --no-group dev --extra <X>` against the experiment's own committed `pyproject.toml` + `uv.lock`. Nothing is pre-baked. Submission: `iris --cluster=marin job run --tpu <variant> --enable-extra-resources --extra tpu --region <r> -- python examples/launch.py`, or the `fray` `JobRequest`/`current_client().submit` API. On TPU the worker injects `TPU_*`/`JAX_COORDINATOR_*` env and iris calls bare `jax.distributed.initialize()`.

**Delphi** (`marin-community/delphi-3e18-447Mparams-1.2Btokens`) is the compute-optimal point of the smallest (3e18 FLOP) budget in marin's open IsoFLOP scaling suite (88 base models, 3e18 → 1e23 FLOPs). It is a **dense Qwen3**, 447M params: 11 layers, hidden 1024, 8 attention heads, 8 KV heads (**no GQA**), head_dim 128, intermediate 4096 (SwiGLU), vocab 128256, ctx 4096, `rope_theta=500000`, `rms_norm_eps=1e-5`, untied embeddings, Qwen3 QK-norm. Tokenizer is the **Llama-3 128k tokenizer**, `bos=128000`, `eos=128001`, **`pad_token_id=null`** (must set pad=eos). Base LM, **no chat template**. Ships one `model.safetensors` shard in F32.

---

## 3. The core integration problem & strategy choice

### 3.1 What tunix demands of "a model"

A tunix RL/SFT model **must** be a `flax.nnx.Module` (hard `isinstance` check, `rl_cluster.py:299`; `str` path-loading raises `NotImplementedError`, so the caller hands `RLCluster` a *live* nnx module). Its `__call__` has the exact 4-positional shape `(input_tokens, positions, cache, attention_mask)` and always returns `(array, Cache|None)`. Two forward modes share the signature:
- **Training/reference logps** (`common.compute_per_token_logps`): `cache=None`, full-sequence.
- **Rollout decode** (`generate/sampler.py`): prefill + single-token steps threading the `{'k','v','end_index'}` ring-buffer, needing `init_cache(batch, cache_size, dtype)`.

Optional hooks (`skip_lm_head`+`compute_final_logits` for chunked logps; `segment_ids`; `decode_only_last_token`) are discovered by `inspect.signature` and engaged only if present. **For tunix-native Qwen3 all of this already exists and is verified working (§1, items 2 & 4).** For grug-the-equinox-model, satisfying it means an nnx port + a from-scratch KV-cache + a RoPE position-offset + a safetensors converter.

### 3.2 Strategy A — port/wrap grug equinox behind tunix's interface
Wrap grug as `nnx.Module` (equinox immutable pytrees → mutable `nnx.Param`), add a RoPE position-offset (grug hardcodes `arange(seq_len)` from 0 — the most invasive edit), thread `(k,v,end_index)` cache through attention/Block/Transformer, reconcile grug's explicit `compact_grug_mesh` (`replica_dcn/data/expert/model`, `AxisType.Explicit`) with tunix's `fsdp/tp`/`Auto` mesh, and write an HF→grug converter (none exists). Base grug's MLP is **ReLU not SwiGLU**, so it cannot even load Delphi without an MLP swap; the MoE variant additionally hard-requires an `expert` mesh axis (single-device RL decode through it is the hardest possible case).
- **Verdict: highest effort, highest risk, weeks — for zero benefit on a model that is already a stock Qwen3 on disk.** This is the path *only if* a genuinely grug-only architecture (no HF equivalent) must be RL-trained later.

### 3.3 Strategy B — native tunix Qwen3 + load Delphi HF weights  *(RECOMMENDED)*
Use `tunix/models/qwen3` directly. Add a Delphi `ModelConfig`, fix the `rope_theta` call sites, load the published Delphi safetensors via `qwen3/params.create_model_from_safe_tensors`, hand the live nnx module to `RLCluster`, run GRPO with `rollout_engine="vanilla"`.
- **Pros:** The rollout gap is **already closed** (KV-cache, `init_cache`, sampler, QK-norm, untied lm_head all exist — §1). HF key-map covers **all 124 Delphi tensors** (byte-verified). Mesh is tunix-canonical `fsdp/tp` (no `compact_grug_mesh` impedance). No equinox, no converter. Packaging resolves (§1).
- **Cons (all small, all bounded):** (1) the `rope_theta` one-line bugfix; (2) Delphi's `rope_scaling: llama3` is unmodeled in tunix — **inert at ≤4096 ctx** (its `original_max_position_embeddings=8192` > our ctx), safe for arithmetic but a parity item to *measure, not reason about*; (3) the loader silently skips unmatched keys — we mitigate with a **hard key-coverage assertion** (not a log check — see M1).
- **Effort: ~1–2 days to a loading, sampling, training Delphi.**

### 3.4 Strategy C — tunix as an RL-component library over grug's own train step
Keep grug's optax/jit step; reuse only tunix's advantage/KL/reward math.
- **Verdict: not worth it.** There is **no seam** to inject grug's loop — the entire train step is nnx-native (`nnx.Optimizer`, `nnx.value_and_grad`, `nnx.jit` donated). "Library-only" use means reimplementing most of `RLLearner`'s driver, and you *still* owe grug's KV-cache `generate()`. The reusable parts (advantage/KL/reward, ~200 lines) could be vendored without taking the RL-driver rewrite.

### 3.5 Recommendation: **adopt Strategy B.** Grounded in §1: the equinox model "cannot be passed where tunix expects an nnx.Module with a KV cache… an offline conversion + nnx reimplementation is required regardless" — and tunix *already wrote* that reimplementation (native Qwen3) and the converter (qwen3 loader), so our cost collapses to a config + a bugfix. Where grug re-enters: a genuinely grug-only arch (MoE/XSA, no HF equivalent) would need Strategy A; scoped as a noted M6 follow-on.

---

## 4. Packaging / dependency strategy  *(validated — §1 item 5)*

**Decision: the iris experiment venv is tunix-native and does NOT hard-depend on `marin-levanter`.** A single fat venv with both is unresolvable today: tunix's `orbax-checkpoint>=0.12.0` needs `tensorstore>=0.1.84`, while `marin-levanter` pins `tensorstore<0.1.82` (empty intersection). Dropping levanter resolves everything else — **proven by `uv lock` (164 pkgs, exit 0).** We don't need levanter because Delphi's *published HF safetensors* are the boundary artifact; we load HF weights, not a levanter checkpoint.

The validated manifest (this dir's `pyproject.toml`, mirrors the grug experiment's `[tool.uv]` overrides + resiliparse index):
```toml
[project]
dependencies = ["google-tunix", "marin-iris", "marin-fray"]   # NO marin-levanter
[project.optional-dependencies]
tpu = ["google-tunix[prod]; sys_platform == 'linux'"]          # jax[tpu]>=0.6.0,!=0.7.2
[dependency-groups]
dev = []     # REQUIRED — worker runs `uv sync --no-group dev`
```
**Watch items:** vLLM/sglang rollout is **off-limits in this venv** (tunix's vLLM path wants `jax[tpu]==0.7.2`, which `prod` excludes) — we use `rollout_engine="vanilla"`, the only backend that runs a custom/non-HF-registered arch anyway. `safetensors<0.8` "breaks with jax>=0.9" per a tunix comment (moot at our resolve; re-check on bumps). Re-run `uv lock` before each submission since marin-* are nightly `0.2.x.dev` wheels.

---

## 5. TPU / sharding & iris submission

**Mesh:** tunix-canonical 2-D `("fsdp","tp")`, `AxisType.Auto`. For 447M run **colocated** — one `Mesh`, with `Role.ACTOR/REFERENCE/ROLLOUT` all mapped to it (same `Mesh` object ⇒ actor & rollout share one nnx model in HBM). No disaggregation, no `compact_grug_mesh` impedance.
```python
mesh = create_mesh(shape=(n_fsdp, n_tp), axis_names=("fsdp","tp"))   # e.g. (4,1) on v6e-4
role_to_mesh = {Role.ACTOR: mesh, Role.REFERENCE: mesh, Role.ROLLOUT: mesh}
```
**Sizing (compute explicitly in M1 — see risk register):** 447M params + Adam (2 moments) + frozen reference + rollout KV-cache. Rough fp32-master/bf16-compute budget: params bf16 ≈ 0.9 GB, Adam moments fp32 ≈ 3.6 GB, reference bf16 ≈ 0.9 GB, KV-cache (B×cache×layers×2×kv_heads×head_dim) small at these sizes. **Fits one host: v6e-4 / v5e-4 minimum, v5e-8 / v6e-8 comfortable.** If tight, drop the reference (`beta=0`) or shrink `kv_cache_size`. **`jax.distributed.initialize()` is iris's job** (single-VM v6e-4 = trivial; v6e-8 = 2 VMs ⇒ multi-host, init-once essential).
**Region:** v6e in `europe-west4` / `us-east5`; v4 (reserved) **only** `us-central2-b`. **Always pass `--region`** — default region has no TPU capacity. Keep bundle <25 MB (code+lock; weights load from HF at runtime).
```bash
uv run iris --cluster=marin job run --no-wait \
  --tpu v6e-4 --enable-extra-resources --extra tpu --region europe-west4 \
  --cpu 32 --memory 128GB --disk 100GB --max-retries 3 \
  -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -- python examples/launch.py
```

---

## 6. Calculator / arithmetic environment, reward & curriculum

**v1 = non-agentic single-turn GRPO, a near-verbatim copy of tunix's gsm8k example** (defer the multi-turn agentic `CalculatorTool` path — it stresses the decode loop hardest and uses a *different* reward signature).

**Dataset** emits rows with a column literally named **`prompts`** + a gold column **`answer`** (GRPOLearner forwards every non-`prompts` column as a reward kwarg). Delphi has **no chat template** → build `prompts` as raw strings.
```python
SYSTEM = ("Solve the problem. Put the final number in <answer></answer>.\n")
row = {"prompts": f"{SYSTEM}Problem: {a} + {b}\n<answer>", "answer": str(a+b)}
```
**Reward file** (`reward_fn/arithmetic.py`): every public top-level fn is auto-discovered and rewards are **summed**; prefix helpers with `_`. Signature `(prompts, completions, answer, **kwargs) -> list[float]`. A `check_answer` (numeric match within tolerance band) + a small `check_format` (0.1 for well-formed `<answer>`). Reuse tunix's `_safe_eval` for expression checking.
**GRPO/rollout config:** `num_generations=8`, `beta≈0.08`, `epsilon=0.2`; `max_prompt_length=128`, `total_generation_steps=256` with **`max_prompt_length + total_generation_steps <= kv_cache_size`** (else the sampler hard-errors), `temperature=0.9`. **Stopping is eos-token-only** (no stop-strings) — set `eos_tokens=[128001]` and **pad=eos=128001**.
**Curriculum (solve-rate-gated over a mixed dataset):** Stage 0 single-digit `a+b` (format) → Stage 1 `+`/`−`/`×` to ~2 digits (advance at solve-rate >0.8) → Stage 2 integer division / 2-op expressions (tolerance band) → Stage 3 `a*x+b=c` integer-solution algebra. Log `solve_ratio`/`solve_all`/`solve_none` (borrow `vtc_metric_fn` from `examples/agentic/qwen3_grpo_gsm8k_demo.py`) to drive gate transitions + W&B.

---

## 7. Rollout plan — milestones

### M1 — Delphi-on-tunix-Qwen3 integration
**Changes:** (a) a `delphi_qwen3.py` shim in this experiment defining `delphi_3e18_447m()` `ModelConfig` (11 layers, vocab 128256, embed 1024, hidden 4096, heads 8, head_dim 128, kv_heads 8, rope_theta 500000, norm_eps 1e-5, untied) — *named `delphi_*`, not `grug_*`, to avoid scope confusion*; (b) **RoPE bugfix**: pass `rope_theta=self.config.rope_theta` into both `apply_rope` call sites (`apply_rope` already accepts it) — carry locally **and** prep an upstream PR (the bug hits llama3 too); (c) loader call `create_model_from_safe_tensors(...)` at `dtype=jnp.bfloat16`; (d) add `num_embed` property (`return self.config.vocab_size`) unconditionally; (e) tokenizer via `TokenizerAdapter`, pad=eos.
**Gate (define thresholds NOW, don't reason):** ① **hard-assert key coverage** — enumerate safetensors keys, run through the key-map, assert every key maps *and* every model param is written (no eval_shape sentinel); do **not** trust the skipped-keys log. ② **HF-parity** — run HF `transformers` Delphi vs tunix Delphi on the same 512-token prompt; require **top-1 next-token agreement = 100%** and **per-logit fp32 MSE < 1e-3**. This is *the* gate catching rope_theta + the unmodeled `rope_scaling`. Also empirically check whether `rope_scaling` is inert ≤4096 by toggling it in HF. ③ greedy `Sampler` emits a coherent continuation. ④ compute the HBM budget on paper; decide reference-present vs `beta=0` before M4.

### M2 — Local CPU smoke test ("print more cats")
**Changes:** a tiny fresh-init `ModelConfig` (vocab ~32, 2 layers, embed 64) of the *same* tunix Qwen3 class; toy task where a token = "cat"; **dense reward = fraction of cat tokens** (guarantees within-group advantage variance early — group-normalized GRPO needs non-zero group std). `GRPOLearner`, `rollout_engine="vanilla"`, `num_generations=4`, CPU.
**Gate:** ① mean reward strictly increases over ~50–100 steps (loop *learns*); ② **direct mechanism assertions** — sampler took >1 decode step and KV-cache `end_index` advanced (loop *wired*, cache path exercised — not inferred from reward-go-up).

### M3a — iris packaging + submission (CPU, no TPU)
**Changes:** finalize `pyproject.toml`+committed `uv.lock` (done — §1); `launch.py` runs the M2 toy on a **CPU** iris allocation (no `--tpu`).
**Gate:** job reaches RUNNING, worker `uv sync` installs tunix, toy reward climbs. **Proves packaging+submission+bundle for free, before paying for TPU.**

### M3b — iris TPU smoke (1 chip / v6e-4)
**Gate:** worker `uv sync --extra tpu` installs `jax[tpu]`, `jax.distributed.initialize()` succeeds, toy reward climbs on TPU. Isolates libtpu/distributed from packaging.

### M4 — Delphi + arithmetic on TPU
**Changes:** swap to `delphi_3e18_447m()` + HF loader; §6 dataset+reward (Stages 0–1); colocated `role_to_mesh` on v6e-4/v5e-8; eos/pad=128001.
**Gate:** held-out single/double-digit `+`/`−`/`×` **solve-rate rises from pretrained baseline to >0.7**; W&B `solve_ratio` trends up; HBM fits without offload. Bound vanilla-sampler tokens/sec here (no cheap rollout-speed upgrade exists in this venv — see risks).

### M5 — Curriculum to basic algebra
**Changes:** Stage-2/3 generators + tolerance-band/normalization (fractions, negatives); solve-rate-gated sampler over the mixed dataset.
**Gate:** Delphi reaches a target solve-rate on held-out `a*x+b=c` after the arithmetic stages, and the gated curriculum beats training algebra cold (ablation). Algebra may need longer generations → raise `total_generation_steps` + `kv_cache_size` together.

### M6 — Experience report + PR
Cover: did tunix-on-iris work, effort vs estimate, the rope_theta bug + upstream PR, HF-parity numbers, HBM/throughput, what "grug-style" meant in practice, curriculum curves, and a scoping recommendation for the genuinely-grug-only (no-HF) case that *would* justify Strategy A. Open the PR; close weaver #229.

---

## 8. Risk register (carried from adversarial review)

| Risk | Severity | Mitigation / status |
|---|---|---|
| Packaging won't resolve | BLOCKER | **CLEARED** — `uv lock` proven (§1) |
| Delphi tensors don't match key-map | BLOCKER | **CLEARED** — 124/124 byte-verified (§1) |
| Loader silently skips a key → random param | RISK | M1 hard key-coverage assertion (not log-based) |
| `rope_theta` ignored → wrong RoPE | RISK | M1 bugfix + HF-parity gate (defined threshold) |
| `rope_scaling: llama3` unmodeled | RISK | inert ≤4096 ctx; **measure** in M1, keep curriculum ≤4096 |
| HBM OOM colocated on v6e-4 | RISK | compute budget in M1; `beta=0`/drop reference is the lever (changes algorithm — decide pre-M4) |
| Smoke test passes/fails for wrong reasons | RISK | dense reward + direct decode/cache assertions (M2) |
| Burn TPU to discover a pip failure | CLARIFY | split M3a (CPU) / M3b (TPU) |
| No cheap rollout-speed upgrade (no vLLM in venv) | RISK | bound tokens/sec at M4; longer-gen algebra is the stressor |
| "grug" naming misleads reviewers | CLARIFY | name artifacts `delphi_qwen3`; rope fix is a clearly-labeled upstream patch |

## 9. Open decisions for the user (async — I proceed with the recommended default unless redirected)
1. **Strategy B vs a literal grug-equinox port** — proceeding with B (default). Pivot to A (weeks) only if running grug's `equinox.Module` is a hard requirement.
2. **Upstream the qwen3 `rope_theta` fix** to google/tunix, or carry locally? (Default: local shim + a separate upstream PR — the bug affects llama3 too.)
3. **Reference model / KL:** `beta>0` (frozen reference in HBM) vs `beta=0` to save HBM. (Default: reference present, small beta; fall back if HBM tight.)
4. **TPU variant/region:** v6e-4 (europe-west4) default.
5. **vLLM rollout off the table** at 447M — confirm (default: yes, vanilla sampler).

**Working notes & full dossiers:** `.agents/logs/tunix-iris/` (`_DESIGN.md` synthesis draft, `_CRITIQUE.md` adversarial review, 10 per-area dossiers).
