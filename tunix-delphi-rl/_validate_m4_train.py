"""M4 validation: run the FULL Delphi arithmetic GRPO pipeline a few steps on CPU.

Proves load -> rollout -> reward -> advantage -> optimizer update wires end-to-end
with the REAL Delphi model and finite loss/reward. CPU is slow (447M); keep steps
tiny. Convergence is deferred to the TPU job.
"""
import time
import jax.numpy as jnp
import numpy as np
from train_delphi import train_delphi_arithmetic

DELPHI_DIR = "/home/power/code/_tunix_lab/delphi"

t0 = time.time()
res = train_delphi_arithmetic(
    model_dir=DELPHI_DIR,
    stage=0,
    steps=3,
    num_generations=4,
    batch_size=2,
    learning_rate=1e-6,
    temperature=0.9,
    max_prompt_length=64,
    max_tokens_to_generate=12,
    beta=0.0,
    seed=0,
)
dt = time.time() - t0
print(f"\n[TRAIN] ran {res.steps_ran} steps in {dt:.1f}s")
print(f"[TRAIN] reward_history    = {res.reward_history}")
print(f"[TRAIN] solve_ratio_history = {res.solve_ratio_history}")
finite = all(np.isfinite(r) for r in res.reward_history)
print(f"[TRAIN] all rewards finite: {finite}")
assert res.steps_ran >= 1, "no steps ran"
assert finite, "non-finite reward encountered"
print("\nM4 TRAIN PIPELINE VALIDATION: PASS (full pipeline wires; finite reward)")
