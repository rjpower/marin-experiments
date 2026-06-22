# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris entrypoint for Delphi CURRICULUM coding RL (issue #8).

Test-case-graded, curriculum-scheduled SFT->Dr.GRPO (see ``CURRICULUM_DESIGN.md``
and :mod:`train_curriculum`). The headline output is per-level pass@1/pass@k on
HELD-OUT instances, before and after RL -- a clear win is RL pass@1 climbing above
the SFT-only control on the mid/high levels.

Config via env vars so the coordinator can size the run:
  * ``CURRIC_TRAIN_LEVELS`` (default ``1,2,3,4,5,6``), ``CURRIC_EVAL_LEVELS`` (= train).
  * ``CURRIC_EVAL_N`` (12)             -- held-out problems per level.
  * ``CURRIC_ROUNDS`` (3)              -- write->test->revise rounds per episode.
  * ``CURRIC_STEPS`` (200)            -- Dr.GRPO steps (0 = SFT/eval only).
  * ``CURRIC_STEPS_PER_LEVEL`` (30)   -- curriculum cadence (steps before unlocking next level).
  * ``CURRIC_PROMOTE_THRESHOLD`` (0.0) -- 0 = pure fixed cadence.
  * ``CURRIC_SFT_STEPS`` (200), ``CURRIC_SFT_LEVELS`` (``1,2``).
  * ``CURRIC_NUM_GENERATIONS`` (16), ``CURRIC_BATCH_SIZE`` (8), ``CURRIC_LR`` (1e-5).
  * ``CURRIC_TEMPERATURE`` (1.0), ``CURRIC_MAX_PROMPT`` (1024), ``CURRIC_MAX_RESPONSE`` (768).
  * ``CURRIC_PASSK`` (16), ``CURRIC_PASSK_TEMP`` (1.0), ``CURRIC_EVAL_TOKENS`` (256).
  * ``CURRIC_MODEL`` (``delphi``) -- registry key; ``CURRIC_MODEL_DIR``.
  * ``CURRIC_CHAT_SFT_STEPS`` (0)  -- Stage-0 "up to shape" chat+tool SFT steps (0 skips).
  * ``CURRIC_CHAT_DATASET`` (``allenai/tulu-3-sft-mixture``), ``CURRIC_CHAT_MIXTURE`` (1).
  * ``CURRIC_CHAT_BATCH_SIZE`` (8), ``CURRIC_CHAT_LR`` (1e-5), ``CURRIC_CHAT_MAX_SEQ_LEN`` (1024).
  * ``CURRIC_SAVE_PATH`` ("")    -- HF-safetensors export dir for the trained actor
    (local or ``gs://``; empty = no export). ``CURRIC_SAVE_DTYPE`` (``bfloat16``).
  * ``CURRIC_SEED`` (0), ``DELPHI_MODEL_DIR`` (``./delphi``).

Submit on a single-host TPU (the coordinator submits; do NOT submit from here):

    uv run iris --cluster=marin job run --no-wait \
      --tpu v6e-4 --enable-extra-resources --extra tpu --region europe-west4 \
      --cpu 8 --memory 64GB --disk 60GB --max-retries 2 --job-name curric-rl \
      -e CURRIC_STEPS 200 -e CURRIC_SFT_STEPS 200 \
      -- python launch_curriculum.py
"""

import os

import jax
from huggingface_hub import snapshot_download

from models.registry import get_model_spec
from training.train_curriculum import train_curriculum


def _ensure_model(repo: str, model_dir: str) -> str:
  if not os.path.exists(os.path.join(model_dir, "config.json")):
    snapshot_download(repo_id=repo, local_dir=model_dir)
  return model_dir


def _parse_levels(raw: str) -> tuple[int, ...]:
  return tuple(int(t) for t in raw.split(",") if t.strip() != "")


def _print_passk(label, pk) -> None:
  if pk is not None:
    print(f"[curric] PASS@K ({label}): {pk.summary()}", flush=True)


def main() -> None:
  train_levels = _parse_levels(os.environ.get("CURRIC_TRAIN_LEVELS", "1,2,3,4,5,6"))
  eval_levels = _parse_levels(os.environ["CURRIC_EVAL_LEVELS"]) if os.environ.get("CURRIC_EVAL_LEVELS") else train_levels
  eval_n = int(os.environ.get("CURRIC_EVAL_N", "12"))
  rounds = int(os.environ.get("CURRIC_ROUNDS", "3"))
  steps = int(os.environ.get("CURRIC_STEPS", "200"))
  steps_per_level = int(os.environ.get("CURRIC_STEPS_PER_LEVEL", "30"))
  promote_threshold = float(os.environ.get("CURRIC_PROMOTE_THRESHOLD", "0.0"))
  sft_steps = int(os.environ.get("CURRIC_SFT_STEPS", "200"))
  sft_levels = _parse_levels(os.environ.get("CURRIC_SFT_LEVELS", "1,2"))
  num_generations = int(os.environ.get("CURRIC_NUM_GENERATIONS", "16"))
  batch_size = int(os.environ.get("CURRIC_BATCH_SIZE", "8"))
  learning_rate = float(os.environ.get("CURRIC_LR", "1e-5"))
  temperature = float(os.environ.get("CURRIC_TEMPERATURE", "1.0"))
  max_prompt = int(os.environ.get("CURRIC_MAX_PROMPT", "1024"))
  max_response = int(os.environ.get("CURRIC_MAX_RESPONSE", "768"))
  passk = int(os.environ.get("CURRIC_PASSK", "16"))
  passk_temp = float(os.environ.get("CURRIC_PASSK_TEMP", "1.0"))
  eval_tokens = int(os.environ.get("CURRIC_EVAL_TOKENS", "256"))
  seed = int(os.environ.get("CURRIC_SEED", "0"))
  chat_sft_steps = int(os.environ.get("CURRIC_CHAT_SFT_STEPS", "0"))
  chat_sft_dataset = os.environ.get("CURRIC_CHAT_DATASET", "allenai/tulu-3-sft-mixture")
  chat_sft_batch_size = int(os.environ.get("CURRIC_CHAT_BATCH_SIZE", "8"))
  chat_sft_lr = float(os.environ.get("CURRIC_CHAT_LR", "1e-5"))
  chat_sft_max_seq_len = int(os.environ.get("CURRIC_CHAT_MAX_SEQ_LEN", "1024"))
  chat_sft_use_mixture = os.environ.get("CURRIC_CHAT_MIXTURE", "1") not in ("0", "false", "False")
  save_path = os.environ.get("CURRIC_SAVE_PATH") or None
  save_dtype = os.environ.get("CURRIC_SAVE_DTYPE", "bfloat16")
  model_name = os.environ.get("CURRIC_MODEL", "delphi")
  model_spec = get_model_spec(model_name)
  model_dir = (
      os.environ.get("CURRIC_MODEL_DIR")
      or os.environ.get("DELPHI_MODEL_DIR")
      or f"./{model_spec.name}"
  )

  print(f"[curric] jax {jax.__version__} devices={jax.devices()}", flush=True)
  print(
      f"[curric] train_levels={train_levels} eval_levels={eval_levels} rounds={rounds} "
      f"steps={steps} steps_per_level={steps_per_level} promote={promote_threshold} "
      f"sft_steps={sft_steps} sft_levels={sft_levels} num_generations={num_generations} "
      f"batch_size={batch_size} lr={learning_rate} temp={temperature} "
      f"max_prompt={max_prompt} max_response={max_response} passk={passk}",
      flush=True,
  )
  print(
      f"[curric] model={model_name} chat_sft_steps={chat_sft_steps} "
      f"chat_sft_mixture={chat_sft_use_mixture} chat_sft_dataset={chat_sft_dataset}",
      flush=True,
  )

  model_dir = _ensure_model(model_spec.repo, model_dir)
  print(f"[curric] model={model_spec.name} repo={model_spec.repo} ready at {model_dir}", flush=True)

  result = train_curriculum(
      model_dir=model_dir,
      train_levels=train_levels,
      eval_levels=eval_levels,
      eval_n_per_level=eval_n,
      rounds=rounds,
      steps=steps,
      steps_per_level=steps_per_level,
      promote_threshold=promote_threshold,
      num_generations=num_generations,
      batch_size=batch_size,
      learning_rate=learning_rate,
      temperature=temperature,
      max_prompt_length=max_prompt,
      max_response_length=max_response,
      seed=seed,
      sft_steps=sft_steps,
      sft_levels=sft_levels,
      passk=passk,
      passk_temperature=passk_temp,
      eval_max_new_tokens=eval_tokens,
      model_spec=model_spec,
      chat_sft_steps=chat_sft_steps,
      chat_sft_dataset=chat_sft_dataset,
      chat_sft_batch_size=chat_sft_batch_size,
      chat_sft_learning_rate=chat_sft_lr,
      chat_sft_max_seq_len=chat_sft_max_seq_len,
      chat_sft_use_mixture=chat_sft_use_mixture,
      save_path=save_path,
      save_dtype=save_dtype,
  )

  for i in range(result.steps_ran):
    def _at(history, idx):
      return history[idx] if idx < len(history) else float("nan")

    print(
        f"[curric] step {i:4d}: reward={_at(result.reward_history, i):.4f} "
        f"first_solve={_at(result.first_solve_history, i):.4f} "
        f"best_solve={_at(result.best_solve_history, i):.4f} "
        f"level={_at(result.level_history, i):.2f}",
        flush=True,
    )

  _print_passk("after-sft" if sft_steps > 0 else "few-shot", result.passk_after_sft)
  _print_passk("after-rl", result.passk_after_rl)
  if save_path:
    print(f"[curric] saved trained model to {save_path}", flush=True)
  print(f"[curric] CURRICULUM COMPLETE (RL steps={result.steps_ran}, train_levels={train_levels})", flush=True)


if __name__ == "__main__":
  main()
