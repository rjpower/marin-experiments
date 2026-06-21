"""Shared training glue: optimizer, device mesh, and the PeftTrainer input fn.

Lifted from the proven ``tunix-delphi-rl`` plumbing. The only generalisation is
:func:`build_mesh`, which now factorizes the local devices into a 2-D
``(fsdp, tp)`` mesh so the 8B actor can use tensor parallelism on a v6e-16 if
pure FSDP runs out of HBM.
"""

import jax
import numpy as np
import optax

from tunix.sft import utils as sft_utils


def clipped_adamw(learning_rate: float) -> optax.GradientTransformation:
  """Global-norm-clipped AdamW (b1=0.9, b2=0.99, wd=0.0).

  The clip is LOAD-BEARING, not a nicety: an occasional exploding update produces
  ``inf``/``NaN`` grads that crash the TPU run with a libtpu ``SIGSEGV``
  mid-training (a hard lesson from the tunix-delphi-rl runs). Clipping the global
  norm to 1.0 bounds the update and keeps the run alive.
  """
  return optax.chain(
      optax.clip_by_global_norm(1.0),
      optax.adamw(learning_rate=learning_rate, b1=0.9, b2=0.99, weight_decay=0.0),
  )


def build_mesh(tp: int = 1) -> jax.sharding.Mesh:
  """Builds a 2-D ``(fsdp, tp)`` mesh over all local devices.

  Args:
    tp: tensor-parallel width. ``tp=1`` gives pure FSDP across every device
      (the default). For an 8B actor on a v6e-16, ``tp=2`` keeps each tensor
      shard larger while still sharding the optimizer state over ``fsdp``.

  Returns:
    A ``jax.sharding.Mesh`` with axis names ``("fsdp", "tp")``. The tunix Qwen3
    ``ShardingConfig`` references both axes, so ``tp>1`` activates tensor
    parallelism with no model-code change.

  Raises:
    ValueError: if ``tp`` does not divide the device count.
  """
  ndev = jax.device_count()
  if ndev % tp != 0:
    raise ValueError(f"tp={tp} does not divide device_count={ndev}.")
  fsdp = ndev // tp
  devices = np.asarray(jax.devices()).reshape(fsdp, tp)
  return jax.sharding.Mesh(devices, axis_names=("fsdp", "tp"))


def sft_model_input_fn(batch: dict) -> dict:
  """Expands a batched SFT row into PeftTrainer ``_default_loss_fn`` kwargs.

  ``input_mask`` is the LOSS mask (which tokens to train on). ``positions`` and
  the ``[B, L, L]`` causal ``attention_mask`` are derived from the separate
  PADDING mask (real tokens vs right-padding), matching the rollout loss path.
  """
  pad_mask = batch["pad_mask"]
  return {
      "input_tokens": batch["input_tokens"],
      "input_mask": batch["loss_mask"],
      "positions": sft_utils.build_positions_from_mask(pad_mask),
      "attention_mask": sft_utils.make_causal_attn_mask(pad_mask),
  }
