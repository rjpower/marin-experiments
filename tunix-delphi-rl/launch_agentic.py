# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris entrypoint for Delphi agentic GRPO (milestone M-port).

Runs the Delphi single-digit-add GRPO loop through tunix's AGENTIC learner
(``train_agentic.train_agentic_port``) on whatever worker iris schedules it on,
proving the single-turn (no-tool) agentic code path on the known-learnable task.
The worker installs this experiment's pinned deps via ``uv sync`` before invoking
``python launch_agentic.py``; this script downloads Delphi's HF weights to a local
worker dir, applies the worker-shippable rope monkeypatch (inside ``load_delphi``),
and runs agentic GRPO.

Run config is read from env vars so the coordinator can size the TPU run:
  * ``DELPHI_AGENT_MODE`` (default ``port``) -- M-port runs the single-turn
    no-tool port. Later milestones (T0+) add tool modes here.
  * ``DELPHI_STEPS`` (default 200)
  * ``DELPHI_STAGE`` (default 0 -- single-digit add, the M-port task)
  * ``DELPHI_NUM_GENERATIONS`` (default 8)
  * ``DELPHI_BATCH_SIZE`` (default 8)
  * ``DELPHI_LR`` (default 1e-5)
  * ``DELPHI_USE_ROLLOUT_LOGPS`` (default 0; set 1 to log the logp_diff canary)
  * ``DELPHI_MODEL_DIR`` (default ``./delphi`` on the worker)

Submit on a single-host TPU (the coordinator submits; do NOT submit from here):

    uv run iris --cluster=marin job run --no-wait \
      --tpu v6e-4 --enable-extra-resources --extra tpu --region europe-west4 \
      --cpu 32 --memory 128GB --disk 100GB --max-retries 3 \
      -e HF_TOKEN "$HF_TOKEN" -e DELPHI_AGENT_MODE port -e DELPHI_STEPS 200 \
      -e DELPHI_STAGE 0 -- python launch_agentic.py
"""

import os

import jax
from huggingface_hub import snapshot_download

from train_agentic import train_agentic_port

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
  """Downloads Delphi and runs agentic arithmetic GRPO on the iris worker."""
  mode = os.environ.get("DELPHI_AGENT_MODE", "port")
  steps = int(os.environ.get("DELPHI_STEPS", "200"))
  stage = int(os.environ.get("DELPHI_STAGE", "0"))
  num_generations = int(os.environ.get("DELPHI_NUM_GENERATIONS", "8"))
  batch_size = int(os.environ.get("DELPHI_BATCH_SIZE", "8"))
  learning_rate = float(os.environ.get("DELPHI_LR", "1e-5"))
  use_rollout_logps = os.environ.get("DELPHI_USE_ROLLOUT_LOGPS", "0") == "1"
  model_dir = os.environ.get("DELPHI_MODEL_DIR", "./delphi")

  if mode != "port":
    raise ValueError(
        f"DELPHI_AGENT_MODE={mode!r} is not supported yet. M-port only runs the "
        "single-turn no-tool 'port' mode; tool modes (T0+) are added later."
    )

  print(f"[launch] jax {jax.__version__} devices={jax.devices()}", flush=True)
  print(
      f"[launch] mode={mode} steps={steps} stage={stage} "
      f"num_generations={num_generations} batch_size={batch_size} "
      f"lr={learning_rate} use_rollout_logps={use_rollout_logps}",
      flush=True,
  )

  model_dir = _ensure_delphi(model_dir)
  print(f"[launch] Delphi ready at {model_dir}", flush=True)

  result = train_agentic_port(
      model_dir=model_dir,
      stage=stage,
      steps=steps,
      num_generations=num_generations,
      batch_size=batch_size,
      learning_rate=learning_rate,
      use_rollout_logps=use_rollout_logps,
  )

  if not result.reward_history:
    raise RuntimeError("Delphi agentic GRPO produced no reward history")

  for i, (reward, solve) in enumerate(
      zip(result.reward_history, result.solve_ratio_history)
  ):
    print(
        f"[launch] step {i:4d}: mean_reward={reward:.4f} "
        f"solve_ratio={solve:.4f}",
        flush=True,
    )

  if result.logp_diff_history:
    last_logp_diff = result.logp_diff_history[-1]
    print(
        f"[launch] LOGP_DIFF (sampler vs trainer, last step): "
        f"{last_logp_diff:.5f} (should be small)",
        flush=True,
    )

  first_solve = result.solve_ratio_history[0] if result.solve_ratio_history else 0.0
  last_solve = result.solve_ratio_history[-1] if result.solve_ratio_history else 0.0
  print(
      f"[launch] SOLVE_RATIO: first={first_solve:.4f} last={last_solve:.4f} "
      f"delta={last_solve - first_solve:+.4f}",
      flush=True,
  )
  print(
      f"[launch] M-PORT COMPLETE ({result.steps_ran} steps, mode={mode})",
      flush=True,
  )


if __name__ == "__main__":
  main()
