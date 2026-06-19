"""M5 validation: run the FULL Delphi arithmetic GRPO pipeline for stages 2 & 3.

Proves load -> rollout -> reward -> advantage -> optimizer update wires end-to-end
on the REAL Delphi model with FINITE reward for each NEW curriculum stage. CPU is
slow (447M); keep steps tiny. Convergence is deferred to the TPU job.

The default ``max_prompt_length=128`` in ``train_delphi_arithmetic`` already
covers stage-3 algebra prompts (~98 tokens); this confirms the pipeline does not
hardcode a stage-0-sized budget.

Run::

    JAX_PLATFORMS=cpu .venv/bin/python _validate_m5_train.py
"""

import time

import numpy as np

from train_delphi import train_delphi_arithmetic

DELPHI_DIR = "/home/power/code/_tunix_lab/delphi"


def _run_stage(stage: int) -> None:
  """Runs a few CPU GRPO steps for ``stage`` and asserts finite reward."""
  t0 = time.time()
  res = train_delphi_arithmetic(
      model_dir=DELPHI_DIR,
      stage=stage,
      steps=3,
      num_generations=4,
      batch_size=2,
      learning_rate=1e-5,
      temperature=0.9,
      max_prompt_length=128,  # stage-3 prompts are ~98 tokens
      max_tokens_to_generate=16,
      beta=0.0,
      seed=0,
  )
  dt = time.time() - t0
  print(f"\n[TRAIN stage {stage}] ran {res.steps_ran} steps in {dt:.1f}s")
  print(f"[TRAIN stage {stage}] reward_history      = {res.reward_history}")
  print(f"[TRAIN stage {stage}] solve_ratio_history = {res.solve_ratio_history}")
  finite = all(np.isfinite(r) for r in res.reward_history)
  print(f"[TRAIN stage {stage}] all rewards finite: {finite}")
  assert res.steps_ran >= 1, f"stage {stage}: no steps ran"
  assert finite, f"stage {stage}: non-finite reward encountered"


def main() -> None:
  for stage in (2, 3):
    _run_stage(stage)
  print("\nM5 TRAIN PIPELINE VALIDATION (stages 2,3): PASS (wires; finite reward)")


if __name__ == "__main__":
  main()
