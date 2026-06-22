"""Load a *stock* HuggingFace Qwen3 base model into tunix's native flax.nnx Qwen3.

This is the general-purpose sibling of :mod:`models.delphi_qwen3`. Delphi is a
quirky Qwen3 (``rope_theta=500000`` + Llama-3 rope scaling, untied embeddings,
Llama-3 tokenizer) that needs a RoPE monkeypatch. A *standard* Qwen3 release
(e.g. ``Qwen/Qwen3-1.7B-Base``) has ``rope_theta=1_000_000`` (== tunix's
``apply_rope`` default) and **no** rope scaling, so the stock tunix RoPE is
already exact and we must NOT apply the Delphi patch.

What this module does:
  * :func:`qwen3_config_from_hf` -- read the HF ``config.json`` and build the
    exact tunix ``ModelConfig`` (handles GQA, depth, tied embeddings, eps, etc.).
  * :func:`load_qwen3` -- the tunix safetensors loader plus the same hard
    key-coverage + all-params-concrete assertions used for Delphi, generalised
    to any Qwen3 config (tied embeddings => no ``lm_head.weight`` key).
  * :func:`load_qwen3_tokenizer` -- the HF tokenizer with pad set to eos.
  * :func:`qwen3_eos_id` -- the tokenizer's eos id (Qwen3 base: 151643).

IMPORTANT: do not load a Delphi model and a stock Qwen3 model in the SAME process
-- :func:`models.delphi_qwen3.load_delphi` monkeypatches the global tunix
``apply_rope`` to Delphi's rope, which would then corrupt a stock Qwen3 forward
pass. One model family per process.
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


def qwen3_config_from_hf(
    model_dir: str,
    *,
    dtype: jnp.dtype = jnp.bfloat16,
    param_dtype: jnp.dtype | None = None,
) -> qm.ModelConfig:
  """Builds the tunix ``ModelConfig`` from a stock Qwen3 ``config.json``.

  Maps HF field names to tunix's: ``hidden_size``->``embed_dim``,
  ``intermediate_size``->``hidden_dim``, ``num_hidden_layers``->``num_layers``,
  ``num_key_value_heads``->``num_kv_heads`` (GQA), ``rms_norm_eps``->``norm_eps``,
  ``tie_word_embeddings``->``use_tied_embedding``.

  Args:
    model_dir: snapshot dir containing ``config.json``.
    dtype: compute dtype for activations.
    param_dtype: storage dtype for params (defaults to ``dtype``).

  Returns:
    A ``qm.ModelConfig`` matching the checkpoint.

  Raises:
    ValueError: if the config is not a Qwen3 architecture, or declares a
      ``rope_scaling`` (stock tunix qwen3 has no rope-scaling field; such a model
      would need the Delphi-style monkeypatch, not this loader).
  """
  with open(os.path.join(model_dir, "config.json"), "r") as f:
    c = json.load(f)

  arch = c.get("architectures") or []
  if c.get("model_type") != "qwen3" and "Qwen3ForCausalLM" not in arch:
    raise ValueError(
        f"qwen3_config_from_hf expects a Qwen3 model; got model_type="
        f"{c.get('model_type')!r}, architectures={arch!r}."
    )
  if c.get("rope_scaling"):
    raise ValueError(
        "config declares rope_scaling="
        f"{c['rope_scaling']!r}; stock tunix qwen3 cannot express it. This loader "
        "is only for un-scaled Qwen3 (use a delphi_patch-style monkeypatch "
        "otherwise)."
    )

  head_dim = c.get("head_dim") or (c["hidden_size"] // c["num_attention_heads"])
  return qm.ModelConfig(
      num_layers=c["num_hidden_layers"],
      vocab_size=c["vocab_size"],
      embed_dim=c["hidden_size"],
      hidden_dim=c["intermediate_size"],
      num_heads=c["num_attention_heads"],
      head_dim=head_dim,
      num_kv_heads=c["num_key_value_heads"],
      rope_theta=int(c.get("rope_theta", 1_000_000)),
      norm_eps=float(c.get("rms_norm_eps", 1e-6)),
      use_tied_embedding=bool(c.get("tie_word_embeddings", False)),
      dtype=dtype,
      param_dtype=param_dtype or dtype,
  )


def _safetensors_keys(model_dir: str) -> list[str]:
  """Reads tensor names from a (possibly sharded) safetensors checkpoint header.

  Reads only the JSON headers, never the tensor data. Handles both a single
  ``model.safetensors`` and a sharded ``model-XXXXX-of-YYYYY.safetensors`` set.

  Args:
    model_dir: dir containing the safetensors file(s).

  Returns:
    The list of tensor names across all shards.
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

  The tunix loader only *logs* a warning on an unmapped key, which would silently
  leave a param at random init. We enumerate the checkpoint keys and run the same
  config-dependent key-map ourselves, raising on any miss.

  Args:
    model_dir: dir with the safetensors checkpoint.
    config: the tunix ``ModelConfig`` (the key-map depends on it -- e.g. tied
      embeddings change which keys are expected).

  Returns:
    The list of checkpoint tensor names (for reporting).

  Raises:
    ValueError: if any checkpoint key fails to map to a model param.
  """
  keys = _safetensors_keys(model_dir)
  key_map = qp._get_key_and_transform_mapping(config)
  unmapped = []
  for k in keys:
    try:
      torch_key_to_jax_key(key_map, k)
    except ValueError:
      unmapped.append(k)
  if unmapped:
    raise ValueError(
        f"{len(unmapped)}/{len(keys)} safetensors keys did not map via the qwen3 "
        f"key-map. Unmapped: {unmapped}"
    )
  return keys


def _assert_all_params_concrete(model: qm.Qwen3) -> None:
  """Asserts no model param still holds an abstract ``eval_shape`` sentinel.

  Any leaf that stayed a ``jax.ShapeDtypeStruct`` means a param was never written
  from the checkpoint.

  Args:
    model: the merged Qwen3 model from the loader.

  Raises:
    ValueError: if any param leaf is not a concrete ``jax.Array``.
  """
  _, state = nnx.split(model)
  pure = state.to_pure_dict()
  abstract = []

  def _check(path, leaf):
    if isinstance(leaf, jax.ShapeDtypeStruct) or not isinstance(leaf, jax.Array):
      abstract.append(".".join(str(getattr(p, "key", p)) for p in path))
    return leaf

  jax.tree_util.tree_map_with_path(_check, pure)
  if abstract:
    raise ValueError(
        f"{len(abstract)} params never written from checkpoint: {abstract}"
    )


def load_qwen3(
    model_dir: str,
    *,
    mesh: jax.sharding.Mesh | None = None,
    dtype: jnp.dtype = jnp.bfloat16,
) -> qm.Qwen3:
  """Loads a stock Qwen3 checkpoint into a tunix Qwen3 with hard coverage checks.

  Unlike :func:`models.delphi_qwen3.load_delphi`, this applies NO RoPE
  monkeypatch: stock tunix ``apply_rope`` (theta default 1e6, no scaling) is
  already correct for a standard Qwen3 with ``rope_theta=1e6`` and no
  ``rope_scaling``. :func:`qwen3_config_from_hf` raises if the checkpoint needs
  scaling.

  Args:
    model_dir: snapshot dir with ``config.json`` + ``*.safetensors``.
    mesh: optional device mesh for sharding the loaded params.
    dtype: dtype to cast loaded params to.

  Returns:
    A live ``qm.Qwen3`` nnx module, fully populated.

  Raises:
    ValueError: on incomplete key coverage or any param left abstract.
  """
  config = qwen3_config_from_hf(model_dir, dtype=dtype, param_dtype=dtype)
  _assert_key_coverage(model_dir, config)
  model = qp.create_model_from_safe_tensors(
      file_dir=model_dir, config=config, mesh=mesh, dtype=dtype
  )
  _assert_all_params_concrete(model)
  return model


def load_qwen3_tokenizer(model_dir: str) -> AutoTokenizer:
  """Loads a Qwen3 HF tokenizer with pad set to eos if unset.

  Args:
    model_dir: dir containing the tokenizer files.

  Returns:
    The HF tokenizer with ``pad_token`` ensured.
  """
  tokenizer = AutoTokenizer.from_pretrained(model_dir)
  if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
  return tokenizer


def qwen3_eos_id(tokenizer) -> int:
  """Returns the tokenizer's eos token id (Qwen3 base default: 151643)."""
  return int(tokenizer.eos_token_id)
