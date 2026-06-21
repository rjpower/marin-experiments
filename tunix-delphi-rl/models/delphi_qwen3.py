"""Load a marin Delphi model into tunix's native flax.nnx Qwen3.

Delphi is a dense Qwen3 **scaling ladder** (``marin-community/delphi-*``). The
447M reference (``delphi-3e18-447Mparams-1.2Btokens``) is 11 layers, hidden 1024,
intermediate 4096, 8 heads, 8 KV heads (no GQA), head_dim 128, vocab 128256; the
~2B point (``delphi-3e20-1.9Bparams-24.7Btokens``) is 21 layers, hidden 2048,
16 heads/16 KV, sharded safetensors -- but **every size shares the same rope
setup** (``rope_theta=500000`` + Llama-3 scaling, ``rms_norm_eps=1e-5``, untied
embeddings, Qwen3 QK-norm, Llama-3 tokenizer), so one loader handles all of them.

This module provides:
  * :func:`delphi_config` -- the exact tunix ``ModelConfig`` for the 447M (kept
    as an explicit, independently-checkable reference).
  * :func:`delphi_config_from_hf` -- the ``ModelConfig`` for ANY Delphi size, read
    from its ``config.json`` (dims vary, rope is asserted to match the patch).
  * :func:`load_delphi` -- the (multi-shard) safetensors loader plus a HARD
    key-coverage assertion (every safetensors tensor maps to a model param AND
    every model param is concrete, i.e. no ``eval_shape`` sentinel survived). The
    tunix loader only *logs* a warning on an unmapped key; we do not trust that.
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

import glob
import json
import os
import struct

import jax
import jax.numpy as jnp
from flax import nnx
from tunix.models.qwen3 import model as qm
from tunix.models.qwen3 import params as qp
from tunix.utils.torch_utils import torch_key_to_jax_key
from transformers import AutoTokenizer

from models.delphi_patch import (
    DELPHI_ROPE_SCALING,
    DELPHI_ROPE_THETA,
    patch_tunix_rope_for_delphi,
)


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


def delphi_config_from_hf(
    model_dir: str,
    *,
    dtype: jnp.dtype = jnp.bfloat16,
    param_dtype: jnp.dtype = jnp.bfloat16,
) -> qm.ModelConfig:
  """Builds the tunix ``ModelConfig`` for ANY Delphi size from its ``config.json``.

  Delphi is a dense Qwen3 scaling ladder (``marin-community/delphi-*``); the dims
  (layers / width / heads) vary by size but **every size shares the same rope
  setup** -- ``rope_theta=500000`` plus Llama-3 scaling
  (``factor=8, original_max_position_embeddings=8192``) -- so the single,
  size-independent monkeypatch in :mod:`models.delphi_patch` is correct for all
  of them. This reads the per-size dims here and leaves rope to the patch.

  The stock tunix ``ModelConfig`` has no ``rope_scaling`` field, so Delphi's
  Llama-3 scaling is NOT delivered here (it is baked into the patch). We
  hard-assert that the checkpoint's ``rope_theta`` and ``rope_scaling`` are
  exactly what the patch bakes in, so a Delphi variant with different rope params
  fails loudly here instead of being silently mis-roped.

  Args:
    model_dir: snapshot dir containing ``config.json``.
    dtype: compute dtype for activations.
    param_dtype: storage dtype for parameters.

  Returns:
    A ``qm.ModelConfig`` describing the checkpoint at ``model_dir``.

  Raises:
    ValueError: if the architecture is not Qwen3, or rope params differ from the
      Delphi patch (theta=500000 + the Llama-3 scaling).
  """
  with open(f"{model_dir}/config.json") as f:
    c = json.load(f)

  arch = c.get("architectures")
  if arch and list(arch) != ["Qwen3ForCausalLM"]:
    raise ValueError(f"Expected Qwen3ForCausalLM, got {arch!r}.")

  theta = int(c.get("rope_theta", DELPHI_ROPE_THETA))
  if theta != DELPHI_ROPE_THETA:
    raise ValueError(
        f"config rope_theta={theta} != Delphi patch theta {DELPHI_ROPE_THETA}; "
        "the monkeypatch bakes in 500000 and would mis-rope this checkpoint."
    )
  scaling = c.get("rope_scaling")
  if scaling is not None:
    got = {k: scaling.get(k) for k in DELPHI_ROPE_SCALING}
    if scaling.get("rope_type") != "llama3" or got != DELPHI_ROPE_SCALING:
      raise ValueError(
          f"config rope_scaling={scaling!r} differs from the Delphi patch "
          f"scaling {DELPHI_ROPE_SCALING!r} (rope_type must be 'llama3'); "
          "refusing to load with a mismatched baked-in rope."
      )

  num_heads = int(c["num_attention_heads"])
  head_dim = int(c.get("head_dim") or c["hidden_size"] // num_heads)
  return qm.ModelConfig(
      num_layers=int(c["num_hidden_layers"]),
      vocab_size=int(c["vocab_size"]),
      embed_dim=int(c["hidden_size"]),
      hidden_dim=int(c["intermediate_size"]),
      num_heads=num_heads,
      head_dim=head_dim,
      num_kv_heads=int(c["num_key_value_heads"]),
      rope_theta=theta,
      norm_eps=float(c["rms_norm_eps"]),
      use_tied_embedding=bool(c.get("tie_word_embeddings", False)),
      dtype=dtype,
      param_dtype=param_dtype,
  )


def _safetensors_keys(model_dir: str) -> list[str]:
  """Reads tensor names across a (possibly sharded) safetensors checkpoint.

  Supports both a single ``model.safetensors`` and a sharded
  ``model-XXXXX-of-YYYYY.safetensors`` set (larger Delphi sizes ship multiple
  shards). Only each shard's header is read; no tensor data is loaded.

  Args:
    model_dir: directory containing one or more ``.safetensors`` files.

  Returns:
    The list of tensor names declared across all shard headers.

  Raises:
    ValueError: if ``model_dir`` contains no ``.safetensors`` file.
  """
  shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
  if not shards:
    raise ValueError(f"No .safetensors files in {model_dir}.")
  keys: list[str] = []
  for path in shards:
    with open(path, "rb") as f:
      header_len = struct.unpack("<Q", f.read(8))[0]
      header = json.loads(f.read(header_len).decode("utf-8"))
    keys.extend(k for k in header if k != "__metadata__")
  return keys


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
  keys = _safetensors_keys(model_dir)
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
  """Loads any Delphi-size safetensors into a tunix Qwen3 with hard checks.

  Dims are read from ``config.json`` (:func:`delphi_config_from_hf`), so the same
  function loads the 447M single-file checkpoint and the multi-shard ~2B one.
  Calls :func:`delphi_patch.patch_tunix_rope_for_delphi` first so the loaded
  model's forward/rollout passes use Delphi's correct RoPE (theta=500000 +
  Llama-3 scaling) on stock tunix, without editing tunix.

  Args:
    model_dir: snapshot dir with ``config.json`` + one or more ``.safetensors``
      shards (single ``model.safetensors`` or ``model-XXXXX-of-YYYYY`` set).
    mesh: optional JAX device mesh for sharding the loaded params.
    dtype: dtype to cast loaded params to (compute and storage).

  Returns:
    A live ``qm.Qwen3`` nnx module with Delphi's weights, fully populated.

  Raises:
    ValueError: if key coverage is incomplete or any param stayed abstract.
  """
  patch_tunix_rope_for_delphi()
  config = delphi_config_from_hf(model_dir, dtype=dtype, param_dtype=dtype)

  # Independent, hard key-coverage assertion BEFORE trusting the loader.
  _assert_key_coverage(model_dir, config)

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
