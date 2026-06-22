"""Worker-shippable RoPE fix for Delphi on STOCK google-tunix 0.1.7.

The iris worker installs stock ``google-tunix`` from PyPI, whose
``tunix.models.qwen3.model.apply_rope(inputs, positions, head_dim,
rope_theta=1_000_000)`` (a) hardcodes ``rope_theta`` at the call sites (it is
never passed, so the 1e6 default is always used — but Delphi needs 500000) and
(b) has no Llama-3 RoPE frequency scaling at all. Delphi
(``marin-community/delphi-3e18-447Mparams-1.2Btokens``) is a Qwen3 with
``rope_theta=500000`` AND ``rope_scaling: {rope_type: llama3, factor: 8, ...}``.
Without both, tunix logits diverge from HuggingFace (M1 measured ~99.7% top-1
agreement / ~5e-3 logit MSE without the scaling, vs 100% / ~7e-12 with it).

M1 delivered this fix by EDITING a local tunix clone (adding a ``rope_scaling``
``ModelConfig`` field, ``_llama3_scale_inv_freq``, and threading ``rope_theta``
+ ``rope_scaling`` through ``apply_rope`` and its call sites). That edit cannot
ship to the worker. This module delivers the *same* numeric fix without touching
tunix: :func:`patch_tunix_rope_for_delphi` monkeypatches
``tunix.models.qwen3.model.apply_rope`` with a drop-in replacement matching the
stock signature, but with Delphi's rope params (theta=500000 + Llama-3 scaling)
BAKED IN. Because the stock Attention call sites invoke ``apply_rope`` without
passing ``rope_theta``, the replacement's baked-in defaults fully control rope.

This is intentionally Delphi-specific: this experiment only ever runs Delphi, so
a baked-in replacement is the clean, minimal worker-shippable delivery.
"""

import jax.numpy as jnp
import jaxtyping
from tunix.models.qwen3 import model as qm


# Delphi's RoPE base frequency (from config.json ``rope_theta``).
DELPHI_ROPE_THETA = 500000

# Delphi's Llama-3 RoPE scaling (from config.json ``rope_scaling``). This is NOT
# inert at short context: it transforms inverse frequencies by WAVELENGTH (not
# sequence length), changing most frequency components at every position, and is
# required for exact HuggingFace parity.
DELPHI_ROPE_SCALING = {
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}


def _llama3_scale_inv_freq(
    inv_freq: jnp.ndarray,
    factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    original_max_position_embeddings: int,
) -> jnp.ndarray:
  """Applies Llama-3 piecewise RoPE frequency scaling.

  Mirrors HuggingFace ``_compute_llama3_parameters`` (and M1's clone edit):
  low-frequency (long-wavelength) components are divided by ``factor``,
  high-frequency components are left unscaled, and a smooth interpolation is
  applied in between. The thresholds are defined by WAVELENGTH (derived from
  ``original_max_position_embeddings`` and the ``*_freq_factor`` values), so the
  scaling is active independent of the runtime sequence length.

  Args:
    inv_freq: base inverse frequencies, shape ``[head_dim // 2]``.
    factor: overall scaling divisor for low-frequency components.
    low_freq_factor: divides the original context to set the low-freq wavelength.
    high_freq_factor: divides the original context to set the high-freq
      wavelength.
    original_max_position_embeddings: the pre-scaling training context length.

  Returns:
    The Llama-3 scaled inverse frequencies, shape ``[head_dim // 2]``.
  """
  low_freq_wavelen = original_max_position_embeddings / low_freq_factor
  high_freq_wavelen = original_max_position_embeddings / high_freq_factor
  wavelen = 2 * jnp.pi / inv_freq

  inv_freq_llama = jnp.where(
      wavelen > low_freq_wavelen, inv_freq / factor, inv_freq
  )
  smooth_factor = (
      original_max_position_embeddings / wavelen - low_freq_factor
  ) / (high_freq_factor - low_freq_factor)
  smoothed_inv_freq = (
      1 - smooth_factor
  ) / factor * inv_freq + smooth_factor * inv_freq
  is_medium_freq = jnp.logical_and(
      wavelen <= low_freq_wavelen, wavelen >= high_freq_wavelen
  )
  return jnp.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)


def _delphi_apply_rope(
    inputs: jaxtyping.Array,  # [B, L, N, head_dim]
    positions: jaxtyping.Array,  # [B, L]
    head_dim: int,
    rope_theta: int = DELPHI_ROPE_THETA,
) -> jaxtyping.Array:
  """Drop-in ``apply_rope`` replacement with Delphi's rope params baked in.

  Matches the stock ``apply_rope(inputs, positions, head_dim,
  rope_theta=1_000_000)`` signature so it can be substituted positionally at the
  stock Attention call sites. The ``rope_theta`` default is Delphi's 500000
  (the stock call sites never pass it), and Delphi's Llama-3 frequency scaling
  is always applied — so this is the exact RoPE M1's clone edit produced.

  Args:
    inputs: query or key projections, shape ``[B, L, N, head_dim]``.
    positions: absolute positions, shape ``[B, L]``.
    head_dim: per-head dimension.
    rope_theta: RoPE base frequency; defaults to Delphi's 500000.

  Returns:
    The rotary-embedded inputs, same shape and dtype as ``inputs``.
  """
  fraction = 2 * jnp.arange(0, head_dim // 2, dtype=jnp.float32) / head_dim
  timescale = rope_theta**fraction

  inv_freq = _llama3_scale_inv_freq(
      1.0 / timescale,
      factor=DELPHI_ROPE_SCALING["factor"],
      low_freq_factor=DELPHI_ROPE_SCALING["low_freq_factor"],
      high_freq_factor=DELPHI_ROPE_SCALING["high_freq_factor"],
      original_max_position_embeddings=DELPHI_ROPE_SCALING[
          "original_max_position_embeddings"
      ],
  )
  timescale = 1.0 / inv_freq

  sinusoid_inp = (
      positions[..., jnp.newaxis] / timescale[jnp.newaxis, jnp.newaxis, :]
  )
  sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :]
  sin = jnp.sin(sinusoid_inp).astype(inputs.dtype)
  cos = jnp.cos(sinusoid_inp).astype(inputs.dtype)

  first_half, second_half = jnp.split(inputs, 2, axis=-1)
  first_part = first_half * cos - second_half * sin
  second_part = second_half * cos + first_half * sin
  out = jnp.concatenate([first_part, second_part], axis=-1)
  return out.astype(inputs.dtype)


def patch_tunix_rope_for_delphi() -> None:
  """Monkeypatches ``tunix.models.qwen3.model.apply_rope`` for Delphi.

  Replaces stock tunix's ``apply_rope`` with :func:`_delphi_apply_rope`, which
  bakes in Delphi's ``rope_theta=500000`` and Llama-3 frequency scaling. This is
  the worker-shippable equivalent of M1's clone edit. Idempotent: re-patching is
  a no-op once the module attribute already points at the replacement.

  The Attention module captures ``apply_rope`` by module-attribute lookup at
  call time (``apply_rope(...)`` inside ``Attention.block``), not by closure, so
  rebinding the module attribute is sufficient — no per-instance patching is
  needed.
  """
  if qm.apply_rope is _delphi_apply_rope:
    return
  qm.apply_rope = _delphi_apply_rope
