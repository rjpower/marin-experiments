# tunix-delphi-rl

A feasibility study: can marin adopt **[google/tunix](https://github.com/google/tunix)** (a JAX-native LLM post-training / RL framework) on **TPUs**, run it through the **[iris](https://github.com/marin-community/marin/tree/main/lib/iris)** cluster manager, and bring a marin model into it? The answer is **yes** — this directory takes it end-to-end, from a toy GRPO smoke test up to the **Delphi** model learning **basic arithmetic and linear algebra** via GRPO on a v6e-4.

The full writeup — feasibility verdict, every change required, integration gotchas, and the ML results — is in [`REPORT.md`](REPORT.md). The up-front design + strategy trade study (A: port grug to nnx; B: load Delphi-as-Qwen3; C: keep equinox) is in [`DESIGN.md`](DESIGN.md).

> **The key reframe.** The target model `marin-community/delphi-3e18-447Mparams-1.2Btokens` is a stock **Qwen3** on disk, and tunix already ships a native `flax.nnx` Qwen3 with a KV-cache sampler and an HF-safetensors loader. So "add a grug-style model to tunix" reduces to a config + a weight loader + one RoPE bug fix — **no `equinox`→`nnx` port, no hand-built generate/KV-cache path.** A genuinely grug-only architecture (MoE GatedNorm/XSA, no HF equivalent) would still need the full port (Strategy A in `DESIGN.md`).

> **Dependencies.** One `uv` venv with `google-tunix` + `marin-iris` + `marin-fray` (164 packages), deliberately **without** `marin-levanter`: tunix's `orbax-checkpoint>=0.12` needs `tensorstore>=0.1.84`, which `marin-levanter` pins below. We don't need levanter — Delphi's *published HF safetensors* are the boundary artifact. The `tpu` extra pulls `google-tunix[prod]` (= `jax[tpu]`); rollout uses the in-process `vanilla` sampler (vLLM/sglang want a different `jax[tpu]` pin and are off-limits in this venv). Repin with `uv lock --upgrade`.

## Layout

Everything is flat in this directory (no package); `launch.py` (toy) and `launch_delphi.py` (Delphi) are the iris entry points.

| file | role |
|---|---|
| `delphi_qwen3.py` | `delphi_config()` (Delphi's exact Qwen3 dims) + `load_delphi()` (safetensors load with a **hard** 124/124 key-coverage assertion) + `load_tokenizer()` |
| `delphi_patch.py` | `patch_tunix_rope_for_delphi()` — the worker-shippable RoPE fix (bakes in `rope_theta=500000` + Llama-3 `rope_scaling`); gives exact HF parity on stock tunix |
| `arithmetic.py` | the "calculator" RL environment: per-stage few-shot dataset (`prompts`/`answer` columns) + `answer_reward` / `proximity_reward` (shaped) / `format_reward` / `metric_fn` (solve_ratio) |
| `train_delphi.py` | `train_delphi_arithmetic(...)` — builds the tunix `RLCluster` + `GRPOLearner` (fp32 actor, vanilla rollout) and runs GRPO |
| `toy_cats.py` | the smoke task: a tiny fresh-init Qwen3 + "emit more `cats`" GRPO |
| `launch.py` / `launch_delphi.py` | iris entrypoints (toy / Delphi); env-driven |
| `test_smoke_cats.py` / `test_delphi_load.py` | M2 learning+wiring gate / M1 HF-parity gate |
| `_validate_m4_*.py`, `_validate_m5_*.py`, `_probe_arith_format.py` | dev-time validation scripts (rope parity, reward correctness, CPU GRPO smoke, base-model format probing) |
| `pyproject.toml`, `uv.lock` | the validated tunix-native manifest |

## Curriculum stages

`arithmetic.py` emits few-shot problems with a stage-appropriate prefix; the model's answer is the first integer it emits after the prompt's marker.

| stage | task | answer space |
|---|---|---|
| 0 | single-digit `a + b` | 0–18 |
| 1 | `a OP b`, `OP∈{+,−,×}`, up to 2 digits | 0–~9800 |
| 2 | two-operation expressions (precedence / parens) | ~0–100+ |
| 3 | linear algebra: `solve for x: a·x + b = c`, integer `x∈[−9,9]` | 19 values |

## Running

CPU smoke test of the toy GRPO loop (no cluster):

```bash
uv sync --frozen --no-group dev
JAX_PLATFORMS=cpu .venv/bin/python test_smoke_cats.py
```

Submit Delphi arithmetic/algebra GRPO onto a TPU on the shared marin cluster (training runs in-process on the job holding the TPU):

```bash
.venv/bin/iris --cluster=marin job run --no-wait \
  --tpu v6e-4 --enable-extra-resources --extra tpu --region europe-west4 \
  --cpu 8 --memory 64GB --disk 60GB --max-retries 5 \
  -e DELPHI_STEPS 400 -e DELPHI_STAGE 0 -e DELPHI_LR 1e-5 -e DELPHI_NUM_GENERATIONS 16 \
  -- python launch_delphi.py
```

`launch_delphi.py` downloads Delphi from HF at runtime (not bundled) and is configured entirely by environment variables:

| env var | meaning | default |
|---|---|---|
| `DELPHI_STAGE` | curriculum stage (0–3) | `0` |
| `DELPHI_STEPS` | GRPO steps | `200` |
| `DELPHI_LR` | actor learning rate (**1e-5** learns; 1e-6 is too low) | `1e-6` |
| `DELPHI_NUM_GENERATIONS` | GRPO group size | `8` |
| `DELPHI_BATCH_SIZE` | prompts per step | `8` |
| `DELPHI_REWARD` | `exact` (answer match) \| `shaped` (proximity partial-credit) | `exact` |
| `DELPHI_MODEL_DIR` | local dir for the downloaded weights | `./delphi` |

## Key results

- **Feasibility: yes, with one upstreamable fix.** tunix installs + runs on iris TPUs with **zero iris changes**; the only tunix change is a RoPE bug fix (`apply_rope` ignored `config.rope_theta` and never applied Llama-3 `rope_scaling` — a latent bug that also affects tunix's llama3). With it, Delphi loads at **exact HF parity** (top-1 100%, logit MSE 7e-12, 124/124 keys).
- **The RL loop learns on TPU.** A toy "emit more cats" task learns 0.08→1.00 on CPU and on a v6e-4; **Delphi learned single-digit addition 6% → ~65%** in ~4 min, and **basic linear algebra (`solve for x`) 4.7% → ~30%** — no calculator tool, plain exact-match reward.
- **RL learnability tracks answer-space *density*, not symbolic difficulty.** Stages with a small answer space (single-digit add, `x∈[−9,9]` algebra) bootstrap from a ~5% base rate and learn; wide-answer-space stages (2-digit, multi-step) sit at ~0% solve rate → zero GRPO advantage → no gradient. Densifying with a proximity reward (`DELPHI_REWARD=shaped`) is the lever for those.
- **GRPO is lr-sensitive on a 447M base model:** lr 1e-6 barely moves; 1e-5 learns cleanly.

Full tables, the friction log, and recommendations are in [`REPORT.md`](REPORT.md). Tracked as weaver issue #229.
