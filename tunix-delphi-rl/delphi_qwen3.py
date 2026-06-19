"""Load the marin Delphi model into tunix's native flax.nnx Qwen3.

Delphi (``marin-community/delphi-3e18-447Mparams-1.2Btokens``) is a dense Qwen3
(447M params): 11 layers, hidden 1024, intermediate 4096, 8 attention heads,
8 KV heads (no GQA), head_dim 128, vocab 128256, ``rope_theta=500000``,
``rms_norm_eps=1e-5``, untied embeddings, Qwen3 QK-norm, Llama-3 tokenizer.

This module provides:
  * :func:`delphi_config` -- the exact tunix ``ModelConfig`` for Delphi.
  * :func:`load_delphi` -- the safetensors loader plus a HARD key-coverage
    assertion (every safetensors tensor maps to a model param AND every model
    param is concrete, i.e. no ``eval_shape`` sentinel survived). The tunix
    loader only *logs* a warning on an unmapped key; we do not trust that.
  * :func:`load_tokenizer` -- the HF Llama-3 tokenizer with pad set to eos.

Stock tunix qwen3 ``apply_rope`` defaults ``rope_theta`` to 1_000_000 and has no
Llama-3 rope scaling, so Delphi's RoPE (theta=500000 + Llama-3 scaling) is wrong
out of the box. Rather than edit tunix (the iris worker installs STOCK tunix
0.1.7 from PyPI), :func:`load_delphi` calls
:func:`delphi_patch.patch_tunix_rope_for_delphi`, which monkeypatches
``apply_rope`` with a drop-in replacement that bakes in Delphi's rope params. The
``rope_theta=500000`` set on the ``ModelConfig`` below is harmless (the
monkeypatch overrides rope entirely); we keep it for documentation. The stock
``ModelConfig`` has NO ``rope_scaling`` field, so we must NOT pass one here.
"""

import json
import struct

import jax
import jax.numpy as jnp
from flax import nnx
from tunix.models.qwen3 import model as qm
from tunix.models.qwen3 import params as qp
from tunix.utils.torch_utils import torch_key_to_jax_key
from transformers import AutoTokenizer

from delphi_patch import patch_tunix_rope_for_delphi


# Delphi vocabulary special tokens (Llama-3 tokenizer, from config.json).
DELPHI_BOS_ID = 128000
DELPHI_EOS_ID = 128001

# The number of tensors in Delphi's single F32 safetensors shard. Used as a
# sanity check on the hard key-coverage assertion.
DELPHI_NUM_TENSORS = 124

def delphi_config(
    *,
    dtype: jnp.dtype = jnp.bfloat16,
    param_dtype: jnp.dtype = jnp.bfloat16,
) -> qm.ModelConfig:
  """Returns the tunix ``ModelConfig`` for the Delphi 447M dense Qwen3.

  Values are taken verbatim from Delphi's ``config.json``. The stock tunix
  ``ModelConfig`` has no ``rope_scaling`` field, so Delphi's Llama-3 rope scaling
  is NOT delivered here; it is delivered via
  :func:`delphi_patch.patch_tunix_rope_for_delphi` (called by :func:`load_delphi`)
  which monkeypatches ``apply_rope`` to bake in both ``rope_theta=500000`` and
  the Llama-3 scaling. That scaling is required for exact HuggingFace parity
  (without it, top-1 next-token agreement is only ~99.7%, MSE ~5e-3; with it,
  agreement is 100%, MSE < 1e-3). The ``rope_theta=500000`` set below is harmless
  (the monkeypatch overrides rope entirely) but kept for documentation.

  Args:
    dtype: compute dtype for activations.
    param_dtype: storage dtype for parameters.

  Returns:
    A ``qm.ModelConfig`` describing Delphi.
  """
  return qm.ModelConfig(
      num_layers=11,
      vocab_size=128256,
      embed_dim=1024,
      hidden_dim=4096,
      num_heads=8,
      head_dim=128,
      num_kv_heads=8,  # no GQA
      rope_theta=500000,
      norm_eps=1e-5,
      use_tied_embedding=False,
      dtype=dtype,
      param_dtype=param_dtype,
  )


def _safetensors_keys(safetensors_path: str) -> list[str]:
  """Reads tensor names from a safetensors file header (no tensor data load).

  Args:
    safetensors_path: path to the ``.safetensors`` file.

  Returns:
    The list of tensor names declared in the file header.
  """
  with open(safetensors_path, "rb") as f:
    header_len = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(header_len).decode("utf-8"))
  return [k for k in header if k != "__metadata__"]


def _assert_key_coverage(model_dir: str, config: qm.ModelConfig) -> list[str]:
  """Asserts every safetensors tensor maps to a model param via the key-map.

  The tunix loader catches an unmapped key and only emits a warning log; a
  silently-skipped key would leave a model param at its random init. We instead
  enumerate the safetensors keys ourselves, run them through the *same* key-map
  the loader uses, and raise on any miss.

  Args:
    model_dir: directory containing ``model.safetensors``.
    config: the Delphi ``ModelConfig`` (the key-map is config-dependent).

  Returns:
    The list of safetensors tensor names (for reporting).

  Raises:
    ValueError: if any safetensors key fails to map to exactly one model param.
  """
  safetensors_path = f"{model_dir}/model.safetensors"
  keys = _safetensors_keys(safetensors_path)
  key_map = qp._get_key_and_transform_mapping(config)

  unmapped: list[str] = []
  for k in keys:
    try:
      torch_key_to_jax_key(key_map, k)
    except ValueError:
      unmapped.append(k)

  if unmapped:
    raise ValueError(
        f"{len(unmapped)}/{len(keys)} safetensors keys did not map to a model "
        f"param via the qwen3 key-map. Unmapped keys: {unmapped}"
    )
  return keys


def _assert_all_params_concrete(model: qm.Qwen3) -> None:
  """Asserts no model param still holds an ``eval_shape`` sentinel.

  The loader builds the model under ``nnx.eval_shape`` (abstract
  ``jax.ShapeDtypeStruct`` leaves) and overwrites each leaf with a concrete
  array as it loads tensors. A tensor that was skipped (never written) would
  leave a ``ShapeDtypeStruct`` behind, so any remaining abstract leaf means a
  param was not populated from the checkpoint.

  Args:
    model: the merged Qwen3 model returned by the loader.

  Raises:
    ValueError: if any param leaf is not a concrete ``jax.Array``.
  """
  _, state = nnx.split(model)
  pure = state.to_pure_dict()
  abstract_paths: list[str] = []

  def _check(path, leaf):
    if isinstance(leaf, jax.ShapeDtypeStruct) or not isinstance(leaf, jax.Array):
      abstract_paths.append(
          ".".join(str(p.key if hasattr(p, "key") else p) for p in path)
      )
    return leaf

  jax.tree_util.tree_map_with_path(_check, pure)

  if abstract_paths:
    raise ValueError(
        f"{len(abstract_paths)} model params were never written from the "
        f"checkpoint (still abstract eval_shape sentinels): {abstract_paths}"
    )


def load_delphi(
    model_dir: str,
    *,
    mesh: jax.sharding.Mesh | None = None,
    dtype: jnp.dtype = jnp.bfloat16,
) -> qm.Qwen3:
  """Loads Delphi safetensors into a tunix Qwen3 with hard coverage checks.

  Calls :func:`delphi_patch.patch_tunix_rope_for_delphi` first so the loaded
  model's forward/rollout passes use Delphi's correct RoPE (theta=500000 +
  Llama-3 scaling) on stock tunix, without editing tunix.

  Args:
    model_dir: directory containing ``model.safetensors`` (the snapshot dir).
    mesh: optional JAX device mesh for sharding the loaded params.
    dtype: dtype to cast loaded params to (compute and storage).

  Returns:
    A live ``qm.Qwen3`` nnx module with Delphi's weights, fully populated.

  Raises:
    ValueError: if key coverage is incomplete or any param stayed abstract.
  """
  patch_tunix_rope_for_delphi()
  config = delphi_config(dtype=dtype, param_dtype=dtype)

  # Independent, hard key-coverage assertion BEFORE trusting the loader.
  keys = _assert_key_coverage(model_dir, config)
  if len(keys) != DELPHI_NUM_TENSORS:
    raise ValueError(
        f"Expected {DELPHI_NUM_TENSORS} Delphi tensors, found {len(keys)}."
    )

  model = qp.create_model_from_safe_tensors(
      file_dir=model_dir,
      config=config,
      mesh=mesh,
      dtype=dtype,
  )

  # Hard assert every param was actually written (no skipped-key fell through).
  _assert_all_params_concrete(model)
  return model


def num_embed(model: qm.Qwen3) -> int:
  """Returns the vocabulary size of a Qwen3 model.

  The tunix Qwen3 module lacks a ``num_embed`` attribute; some sampler paths
  read it only when ``return_logits=True`` (which defaults to ``False``). This
  helper provides it defensively for callers that need it without monkeypatching
  the upstream class.

  Args:
    model: a Qwen3 model.

  Returns:
    The model's vocabulary size.
  """
  return model.config.vocab_size


def load_tokenizer(model_dir: str) -> AutoTokenizer:
  """Loads Delphi's HF Llama-3 tokenizer with pad set to eos.

  Delphi has ``pad_token_id=null`` in its config; tunix's sampler and the
  arithmetic GRPO loop both require a pad token, so we set pad=eos (128001),
  matching the design's eos-only stopping convention.

  Args:
    model_dir: directory containing the tokenizer files.

  Returns:
    The HF tokenizer with ``pad_token`` set to the eos token.
  """
  tokenizer = AutoTokenizer.from_pretrained(model_dir)
  if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
  return tokenizer
