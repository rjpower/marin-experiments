# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris entrypoint for Delphi arithmetic GRPO (milestone M4).

Runs the Delphi arithmetic GRPO loop validated locally in M4-prep
(``train_delphi.train_delphi_arithmetic``) on whatever worker iris schedules it
on. The worker installs this experiment's pinned deps via ``uv sync`` before
invoking ``python launch_delphi.py``; this script then downloads Delphi's HF
weights to a local worker dir (Delphi is ~1.8 GB; the worker has disk), applies
the worker-shippable rope monkeypatch (inside ``load_delphi``), and runs GRPO.

Steps / stage are read from env vars so the coordinator can size the TPU run:
  * ``DELPHI_STEPS`` (default 200)
  * ``DELPHI_STAGE`` (default 0)
  * ``DELPHI_NUM_GENERATIONS`` (default 8)
  * ``DELPHI_BATCH_SIZE`` (default 8)
  * ``DELPHI_LR`` (default 1e-6)
  * ``DELPHI_MODEL_DIR`` (default ``./delphi`` on the worker)

Submit on a single-host TPU (the coordinator submits; do NOT submit from here):

    uv run iris --cluster=marin job run --no-wait \
      --tpu v6e-4 --enable-extra-resources --extra tpu --region europe-west4 \
      --cpu 32 --memory 128GB --disk 100GB --max-retries 3 \
      -e HF_TOKEN "$HF_TOKEN" -e DELPHI_STEPS 200 -e DELPHI_STAGE 0 \
      -- python launch_delphi.py
"""

import os

import jax
from huggingface_hub import snapshot_download

from training.train_delphi import train_delphi_arithmetic

DELPHI_REPO = "marin-community/delphi-3e18-447Mparams-1.2Btokens"


def _ensure_delphi(model_dir: str) -> str:
  """Downloads Delphi to ``model_dir`` on the worker if not already present.

  Args:
    model_dir: local directory to snapshot Delphi into.

  Returns:
    The model directory (containing ``model.safetensors`` + tokenizer files).
  """
  if not os.path.exists(os.path.join(model_dir, "model.safetensors")):
    snapshot_download(repo_id=DELPHI_REPO, local_dir=model_dir)
  return model_dir


def main() -> None:
  """Downloads Delphi and runs arithmetic GRPO on the iris worker."""
  steps = int(os.environ.get("DELPHI_STEPS", "200"))
  stage = int(os.environ.get("DELPHI_STAGE", "0"))
  num_generations = int(os.environ.get("DELPHI_NUM_GENERATIONS", "8"))
  batch_size = int(os.environ.get("DELPHI_BATCH_SIZE", "8"))
  learning_rate = float(os.environ.get("DELPHI_LR", "1e-6"))
  reward_mode = os.environ.get("DELPHI_REWARD", "exact")
  model_dir = os.environ.get("DELPHI_MODEL_DIR", "./delphi")

  print(f"[launch] jax {jax.__version__} devices={jax.devices()}", flush=True)
  print(
      f"[launch] steps={steps} stage={stage} num_generations={num_generations} "
      f"batch_size={batch_size} lr={learning_rate} reward={reward_mode}",
      flush=True,
  )

  model_dir = _ensure_delphi(model_dir)
  print(f"[launch] Delphi ready at {model_dir}", flush=True)

  result = train_delphi_arithmetic(
      model_dir=model_dir,
      stage=stage,
      steps=steps,
      num_generations=num_generations,
      batch_size=batch_size,
      learning_rate=learning_rate,
      reward_mode=reward_mode,
  )

  if not result.reward_history:
    raise RuntimeError("Delphi GRPO produced no reward history")

  for i, (reward, solve) in enumerate(
      zip(result.reward_history, result.solve_ratio_history)
  ):
    print(
        f"[launch] step {i:4d}: mean_reward={reward:.4f} "
        f"solve_ratio={solve:.4f}",
        flush=True,
    )

  first_solve = result.solve_ratio_history[0] if result.solve_ratio_history else 0.0
  last_solve = result.solve_ratio_history[-1] if result.solve_ratio_history else 0.0
  print(
      f"[launch] SOLVE_RATIO: first={first_solve:.4f} last={last_solve:.4f} "
      f"delta={last_solve - first_solve:+.4f}",
      flush=True,
  )
  print(f"[launch] M4 DELPHI ARITHMETIC GRPO COMPLETE ({result.steps_ran} steps)", flush=True)


if __name__ == "__main__":
  main()
