# Serving the §13 RL'd Qwen3-1.7B as an Iris endpoint

**Goal.** Take the §13 Dr.GRPO win (Qwen3-1.7B-Base coding, pass@1 0.29→0.47) and make it
*queryable*: a long-lived HTTP server on an Iris TPU, reachable from anywhere through the
Iris proxy resolver.

## The finding that shaped this

`train_curriculum` **never checkpoints** — it runs SFT→Dr.GRPO in place on one in-memory
`nnx` actor, evaluates it, and returns only pass@k floats (`training/train_curriculum.py`,
and REPORT.md:349). Nothing from the §13 run exists in GCS. So "export the model" is not a
download — the RL'd weights must be **re-created by re-running the recipe with a save step
added**. The served model is a statistically-equivalent re-run of §13.

## Design decisions

- **Inference engine: tunix `Sampler` + FastAPI**, not vLLM. Reuses the exact load+sample
  code that produced the report numbers (`environments/curriculum_env.py:330-364`), the
  existing venv, and needs no vLLM-on-TPU bring-up. (vLLM-on-TPU is real — marin has the
  brokered machinery in `lib/marin/src/marin/inference/vllm.py` — but it is the heavyweight
  path: alpha `tpu-inference`, separate image, multi-minute XLA compiles, untested at 1.7B.)
- **Persistence: HF safetensors**, not orbax. Reload reuses the already-tested
  `models.qwen3_loader.load_qwen3`, which hard-asserts key coverage + concreteness (a
  built-in correctness gate). The artifact is also portable to transformers / vLLM later.
- **Re-train first, then serve** the trained artifact (per request).

## Artifact flow

```
re-run §13 Dr.GRPO  ──save──▶  HF safetensors in GCS  ──reload──▶  serve job (v6e-4)
  (train job, v6e-8)          gs://marin-us-east5/          load_qwen3 + tunix Sampler
  launch_curriculum.py        rl-checkpoints/qwen17-…        + FastAPI HTTP
  + CURRIC_SAVE_PATH                                          └─ register Iris endpoint
                                                                 │
  client ── ProxyResolver ──▶ controller /proxy/<ns>.delphi-rl/generate ──▶ serve
```

## Files

| file | role |
|---|---|
| `serving/export_hf.py` | `save_qwen3_to_hf(model, out_dir, *, hf_config_dir, save_dtype)` — inverts the tunix qwen3 key-map → HF `model.safetensors` + copied `config.json`/tokenizer; `gs://` upload via **gcsfs** (in the locked venv; gsutil is NOT on the worker). |
| `serving/serve.py` | Worker entrypoint: download ckpt (gs→local) → `load_qwen3` → tunix `Sampler` → FastAPI (`/health`, `/generate`, `/v1/completions`) on the Iris-allocated named port → register endpoint → block. Serializes generation under a lock; **always passes `top_p`** (greedy-bug guard). |
| `serving/launch_serve.py` | Submits `serve.py` via `IrisClient.submit(ports=["http"], …)` on a v6e-4. The `iris job run` CLI has **no** named-port flag, so the Python submit path is canonical. |
| `serving/query.py` | Client: `ProxyResolver(controller).resolve("/<ns>/delphi-rl")` → POST `/generate`. |
| `serving/_export_smoke.py` | TPU smoke: load real 1.7B → export → reload → parity (tests sharded `device_get` + worker GCS write). |
| `tests/test_export_hf_roundtrip.py` | CPU gate **G1**: fresh-init Qwen3 → export → reload → logit parity (tied + untied). |
| `tests/test_serve.py` | CPU gate **G2**: FastAPI handlers, cache clamp, top_p-always, proxy URL form. |
| edits: `training/train_curriculum.py`, `launch_curriculum.py` | optional `save_path`/`CURRIC_SAVE_PATH` → export the trained actor after RL. |

## Commands

**Phase 1 — re-train §13 + save** (v6e-8; the launcher default rollout shape OOMs HBM, so the
shrunk shape from REPORT.md:349 is used):

```bash
.venv/bin/iris --cluster=marin job run --no-wait \
  --tpu v6e-8 --enable-extra-resources --extra tpu --region us-east5 \
  --cpu 8 --memory 200GB --disk 80GB --max-retries 3 --job-name qwen17-curric-rl \
  -e CURRIC_MODEL qwen3 -e CURRIC_SFT_STEPS 0 -e CURRIC_STEPS 150 -e CURRIC_ROUNDS 2 \
  -e CURRIC_TRAIN_LEVELS 1,2,3,4,5,6 -e CURRIC_EVAL_LEVELS 1,2,3,4,5,6,7,8,9 \
  -e CURRIC_NUM_GENERATIONS 8 -e CURRIC_BATCH_SIZE 3 \
  -e CURRIC_MAX_PROMPT 896 -e CURRIC_MAX_RESPONSE 512 -e CURRIC_LR 2e-6 \
  -e CURRIC_SAVE_PATH gs://marin-us-east5/rl-checkpoints/qwen17-curric-rl \
  -- python launch_curriculum.py
```

**Phase 2 — serve** (v6e-4 is plenty for 1.7B inference):

```bash
.venv/bin/python serving/launch_serve.py --no-dry-run \
  --controller <controller-url> --region us-east5 \
  --ckpt gs://marin-us-east5/rl-checkpoints/qwen17-curric-rl \
  --endpoint delphi-rl
```

**Phase 3 — query** through the proxy:

```bash
.venv/bin/python serving/query.py --controller <controller-url> \
  --endpoint /power/delphi-rl --prompt "def solve(n):  # sum of primes below n"
```

## Verification gates

| gate | what | status |
|---|---|---|
| G1 | CPU export→reload logit parity (tied + untied) | ✅ `pytest tests/test_export_hf_roundtrip.py` |
| G1b | real-GCS export→reload, **bit-exact** (fp32) | ✅ via gcsfs |
| G2 | FastAPI handlers / clamp / top_p / proxy URL | ✅ `pytest tests/test_serve.py` (12) |
| G3 | TPU export smoke on real 1.7B (sharded gather + worker GCS) | ▶ `/power/export-smoke` |
| G4 | re-run §13 + checkpoint in GCS | ▶ Phase 1 |
| G5 | serve restored ckpt + query via proxy | ▶ Phase 2/3 |

## Gotchas (load-bearing)

- **tunix `Sampler` greedy bug:** decodes greedily and silently ignores `temperature`/`seed`
  unless `top_p` is passed. The server always passes `top_p` (default 1.0).
- **`cache_size` is fixed at Sampler construction** (`max_prompt + max_new + 16`). Requests
  over budget are clamped/rejected.
- **Endpoint visibility is tied to job liveness** — a crashed serve job drops its endpoint;
  submit with generous retries.
- **Namespace prefixing:** `registry.register("delphi-rl", …)` registers `/<ns>/delphi-rl`;
  callers resolve that slash-prefixed full name.
- **GCS write from the worker uses gcsfs** (locked venv), not gsutil/gcloud (not on the
  worker image).
- **Region:** checkpoint bucket `gs://marin-us-east5` is in us-east5; run TPU jobs there
  (v6e lives in us-east5-b) to keep the write in-region.
- **Bundle:** iris ships `git ls-files --cached --others --exclude-standard`, so untracked
  non-ignored files are included (no commit needed); keep the dir under the 25 MB cap.

## What shipped (beyond the original plan)

The plan above is the §13 Qwen coding serve; the implementation generalized to **two serve
task modes** (`SERVE_TASK`) and **three live endpoints**.

- **`task=coding`** (the plan): single-turn `/generate`. Served the §13 RL'd Qwen3-1.7B at
  `/tunix/delphi-rl` — reproduced **pass@1 0.281 → 0.409** (after-RL), exported via the
  `train_curriculum.py` save hook (`CURRIC_SAVE_PATH`).
- **`task=calc`**: a multi-turn **CALC tool-use agent loop** run server-side, mirroring the
  §8 rollout (`generate → CalcTextToolParser → CalculatorTool → inject "Tool result: X" →
  repeat`), sized per stage (`_CALC_STAGES`, t0/t1/t2), stopping on Delphi's
  `newline_terminal_eos_tokens`. New routes: `/calc` + a transcript dashboard with a
  **side-by-side compare mode** (`SERVE_COMPARE_ENDPOINTS`, sibling proxy fetch). `SERVE_CKPT`
  also accepts an `org/repo` HF id (snapshot_download), so raw base models serve with no
  staging. Export hook added to `train_agentic.py` (`DELPHI_SAVE_PATH`) — the qwen3 exporter
  is rope-agnostic, so it works unchanged on the rope-monkeypatched Delphi actor.
- **Delphi base-vs-RL calc demo:** `/tunix/delphi-calc-base` (raw 447M) vs
  `/tunix/delphi-calc-rl` (T1 SFT→Dr.GRPO). On 3-operand chains: **base 0/5, RL 5/5** — the RL
  model chains tool calls correctly; the base botches the operand copy. SFT installs the
  skill (solve_ratio 1.0 from the first GRPO step), GRPO holds it.

Extra calc gotchas: the §8 reward only needs the gold to appear as a standalone int, so a
correct model emits a noisy `"420 * 7"` finish — `serve.py:_clean_final_answer` headlines the
verified product (the last tool result) while the transcript keeps the raw turn. The T1 model
is specialized to 3-operand chains (a 2-operand input is OOD). Calc serve decodes greedily
(`temperature=0.0`, `top_p` OMITTED).

| gate | what | status |
|---|---|---|
| G4 | re-run §13 + checkpoint in GCS | ✅ `gs://marin-us-east5/rl-checkpoints/qwen17-curric-rl` |
| G5 | serve restored ckpt + query via proxy | ✅ `/tunix/delphi-rl` greedy + sampling |
| G6 | Delphi calc SFT→Dr.GRPO export + base-vs-RL serve | ✅ `/tunix/delphi-calc-{base,rl}` (0/5 vs 5/5) |
