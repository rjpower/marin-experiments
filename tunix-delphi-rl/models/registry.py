"""A tiny registry mapping a model name to its HF repo + tunix loaders.

Lets the curriculum stack (training/launch) pick a base model by name instead of
hardcoding Delphi. Both loaders share the signature
``load(model_dir, *, mesh=None, dtype=...) -> nnx model`` and
``load_tokenizer(model_dir) -> HF tokenizer`` so callers are uniform. The eos id
is always taken from ``tokenizer.eos_token_id`` (for Delphi that is the same
128001 the old ``DELPHI_EOS_ID`` constant held), so no per-model eos wiring.

NOTE: never select two different families in one process -- Delphi's loader
monkeypatches the global tunix ``apply_rope`` (see
:mod:`models.qwen3_loader`). One model family per job.
"""

import dataclasses
from typing import Callable


@dataclasses.dataclass(frozen=True)
class ModelSpec:
  """A base model: its display name, HF repo, and tunix loaders."""

  name: str
  repo: str
  load_model: Callable  # (model_dir, *, mesh=None, dtype=...) -> nnx model
  load_tokenizer: Callable  # (model_dir) -> HF tokenizer


def _delphi_spec() -> ModelSpec:
  from models.delphi_qwen3 import load_delphi, load_tokenizer

  return ModelSpec(
      name="delphi",
      repo="marin-community/delphi-3e18-447Mparams-1.2Btokens",
      load_model=load_delphi,
      load_tokenizer=load_tokenizer,
  )


def _qwen3_17b_base_spec() -> ModelSpec:
  from models.qwen3_loader import load_qwen3, load_qwen3_tokenizer

  return ModelSpec(
      name="qwen3-1.7b-base",
      repo="Qwen/Qwen3-1.7B-Base",
      load_model=load_qwen3,
      load_tokenizer=load_qwen3_tokenizer,
  )


# Name -> factory (lazy so importing this module doesn't import every loader).
_REGISTRY: dict[str, Callable[[], ModelSpec]] = {
    "delphi": _delphi_spec,
    "qwen3": _qwen3_17b_base_spec,
    "qwen3-1.7b": _qwen3_17b_base_spec,
    "qwen3-1.7b-base": _qwen3_17b_base_spec,
}


def get_model_spec(name: str = "delphi") -> ModelSpec:
  """Returns the :class:`ModelSpec` for ``name`` (default ``delphi``).

  Args:
    name: a key in the registry (case-insensitive): ``delphi`` or
      ``qwen3`` / ``qwen3-1.7b-base``.

  Returns:
    The resolved :class:`ModelSpec`.

  Raises:
    KeyError: if ``name`` is not registered.
  """
  key = name.strip().lower()
  if key not in _REGISTRY:
    raise KeyError(
        f"Unknown model {name!r}; known: {sorted(_REGISTRY)}."
    )
  return _REGISTRY[key]()
