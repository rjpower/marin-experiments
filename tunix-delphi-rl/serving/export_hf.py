# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export a trained tunix (flax.nnx) Qwen3 model to HuggingFace safetensors.

The result directory is loadable BOTH by the repo's native
:func:`models.qwen3_loader.load_qwen3` AND by ``transformers`` / vLLM, because
it is the standard HF on-disk layout: ``model.safetensors`` (HF tensor names and
shapes) plus the verbatim base-repo ``config.json`` and tokenizer files.

This is the exact inverse of the tunix *load* path. The loader
(:mod:`tunix.models.safetensors_loader`) takes each HF tensor and applies
``transpose(permute)`` then ``reshape(reshape)`` to produce the nnx leaf, where
``(permute, reshape)`` come from
:func:`tunix.models.qwen3.params._get_key_and_transform_mapping`. We enumerate
the nnx leaves and run the inverse (undo reshape, then undo transpose) to
recover the HF tensor, so a roundtrip is bit-exact up to dtype.

Architecture is unchanged by training, so the base ``config.json`` (including
``tie_word_embeddings``) is exactly correct and is copied verbatim. With tied
embeddings there is no ``lm_head`` leaf and we emit no ``lm_head.weight`` key,
relying on the copied config's ``tie_word_embeddings: true``.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile

import jax
import ml_dtypes
import numpy as np
from flax import nnx
import safetensors.numpy

# Tokenizer / config files copied verbatim from the base HF snapshot (whichever
# exist; missing files are tolerated).
_AUX_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
    "added_tokens.json",
    "chat_template.jinja",
)


def _nnx_path_to_key(path) -> str:
  """Builds the dotted leaf key the loader's ``path_to_key`` would build.

  Numeric list indices stay numeric (no quoting), and any trailing ``.value``
  that ``nnx.Param`` introduces is already absent from ``to_pure_dict()``.

  Args:
    path: a jax key-path tuple from ``tree_leaves_with_path``.

  Returns:
    The dotted path, e.g. ``layers.0.attn.q_proj.w``.
  """
  parts = []
  for key in path:
    k = key.key if hasattr(key, "key") else key
    parts.append(str(k))
  return ".".join(parts)


def _to_hf_tensor(nnx_key: str, arr: np.ndarray, cfg) -> tuple[str, np.ndarray]:
  """Maps one nnx leaf (key + host array) to its HF (key, tensor).

  This is the algebraic inverse of the tunix qwen3 load transforms. Load does
  ``hf.transpose(permute).reshape(reshape) == nnx``; we undo it.

  Args:
    nnx_key: dotted nnx leaf path (e.g. ``layers.3.attn.q_proj.w``).
    arr: the leaf value as a host numpy array.
    cfg: the ``tunix.models.qwen3.model.ModelConfig``.

  Returns:
    ``(hf_key, hf_tensor)``.

  Raises:
    NotImplementedError: on an MoE expert / router leaf (dense Qwen3 only).
    ValueError: on an unrecognised leaf path.
  """
  d = cfg.embed_dim
  n = cfg.num_heads
  k = cfg.num_kv_heads
  h = cfg.head_dim

  if nnx_key == "embedder.input_embedding":
    return "model.embed_tokens.weight", arr
  if nnx_key == "final_norm.w":
    return "model.norm.weight", arr
  if nnx_key == "lm_head.w":
    # load: hf (V, D).T -> nnx (D, V); inverse: nnx (D, V).T -> hf (V, D)
    return "lm_head.weight", arr.T

  m = re.fullmatch(r"layers\.([0-9]+)\.(.+)", nnx_key)
  if m is None:
    raise ValueError(f"Unrecognised nnx leaf path: {nnx_key!r}")
  i = m.group(1)
  sub = m.group(2)
  pre = f"model.layers.{i}"

  # Attention projections. Load applies transpose(1,0) then reshape; we undo
  # reshape (back to 2-D) then undo the transpose.
  if sub == "attn.q_proj.w":
    # nnx (D, N, H) -> (D, N*H) -> .T -> hf (N*H, D)
    return f"{pre}.self_attn.q_proj.weight", arr.reshape(d, n * h).T
  if sub == "attn.k_proj.w":
    return f"{pre}.self_attn.k_proj.weight", arr.reshape(d, k * h).T
  if sub == "attn.v_proj.w":
    return f"{pre}.self_attn.v_proj.weight", arr.reshape(d, k * h).T
  if sub == "attn.o_proj.w":
    # nnx (N, H, D) -> (N*H, D) -> .T -> hf (D, N*H)
    return f"{pre}.self_attn.o_proj.weight", arr.reshape(n * h, d).T

  # MLP linears: load is a bare transpose(1,0); inverse is .T.
  if sub == "mlp.gate_proj.kernel":
    return f"{pre}.mlp.gate_proj.weight", arr.T
  if sub == "mlp.up_proj.kernel":
    return f"{pre}.mlp.up_proj.weight", arr.T
  if sub == "mlp.down_proj.kernel":
    return f"{pre}.mlp.down_proj.weight", arr.T

  # Norms (no transform).
  if sub == "attn.q_norm.w":
    return f"{pre}.self_attn.q_norm.weight", arr
  if sub == "attn.k_norm.w":
    return f"{pre}.self_attn.k_norm.weight", arr
  if sub == "input_layernorm.w":
    return f"{pre}.input_layernorm.weight", arr
  if sub == "post_attention_layernorm.w":
    return f"{pre}.post_attention_layernorm.weight", arr

  if "experts" in sub or sub == "mlp.router.kernel":
    raise NotImplementedError(
        f"MoE expert/router leaf {nnx_key!r} is not supported; this exporter "
        "handles only dense Qwen3 (num_experts is None)."
    )

  raise ValueError(f"Unrecognised nnx leaf path: {nnx_key!r}")


def _build_state_dict(model, save_dtype: str) -> dict[str, np.ndarray]:
  """Enumerates nnx leaves and builds the HF safetensors state dict.

  Args:
    model: a live ``tunix.models.qwen3.model.Qwen3``.
    save_dtype: ``"bfloat16"`` or ``"float32"``.

  Returns:
    Mapping of HF tensor name -> host numpy array in ``save_dtype``.

  Raises:
    ValueError: on an unknown ``save_dtype``.
  """
  if save_dtype == "bfloat16":
    np_dtype = ml_dtypes.bfloat16
  elif save_dtype == "float32":
    np_dtype = np.float32
  else:
    raise ValueError(
        f"save_dtype must be 'bfloat16' or 'float32', got {save_dtype!r}."
    )

  cfg = model.config
  _, state = nnx.split(model)
  pure = state.to_pure_dict()

  state_dict: dict[str, np.ndarray] = {}
  for path, leaf in jax.tree_util.tree_leaves_with_path(pure):
    nnx_key = _nnx_path_to_key(path)
    arr = np.asarray(jax.device_get(leaf))
    hf_key, hf_arr = _to_hf_tensor(nnx_key, arr, cfg)
    state_dict[hf_key] = np.ascontiguousarray(hf_arr.astype(np_dtype))
  return state_dict


def _copy_aux_files(hf_config_dir: str, out_dir: str) -> None:
  """Copies config + tokenizer files verbatim, tolerating absent ones."""
  for name in _AUX_FILES:
    src = os.path.join(hf_config_dir, name)
    if os.path.isfile(src):
      shutil.copy(src, os.path.join(out_dir, name))


def _write_local(model, local_dir: str, *, hf_config_dir: str, save_dtype: str) -> None:
  """Writes safetensors + aux files into a local directory."""
  os.makedirs(local_dir, exist_ok=True)
  state_dict = _build_state_dict(model, save_dtype)
  safetensors.numpy.save_file(
      state_dict, os.path.join(local_dir, "model.safetensors")
  )
  _copy_aux_files(hf_config_dir, local_dir)


def save_qwen3_to_hf(
    model,
    out_dir: str,
    *,
    hf_config_dir: str,
    save_dtype: str = "bfloat16",
) -> None:
  """Exports a tunix nnx Qwen3 to a HF safetensors directory.

  Writes ``model.safetensors`` (HF tensor names/shapes), plus a verbatim copy of
  the base repo's ``config.json`` and tokenizer files. The output loads with
  both :func:`models.qwen3_loader.load_qwen3` and ``transformers`` / vLLM.

  Args:
    model: a live ``tunix.models.qwen3.model.Qwen3`` (e.g. an RL-trained actor).
      Its ``model.config`` must be a ``ModelConfig`` and must be dense
      (``num_experts is None``).
    out_dir: local path or a ``gs://...`` path to write to.
    hf_config_dir: local snapshot dir of the BASE HF repo (has ``config.json``
      + tokenizer files). Since training does not change the architecture, this
      base config is exactly correct (including ``tie_word_embeddings``).
    save_dtype: ``"bfloat16"`` (default) or ``"float32"``.

  Raises:
    NotImplementedError: if the model contains MoE expert leaves.
    RuntimeError: if a ``gs://`` upload fails.
  """
  if out_dir.startswith("gs://"):
    with tempfile.TemporaryDirectory() as tmp:
      _write_local(model, tmp, hf_config_dir=hf_config_dir, save_dtype=save_dtype)
      _upload_dir_to_gcs(tmp, out_dir.rstrip("/"))
  else:
    _write_local(
        model, out_dir, hf_config_dir=hf_config_dir, save_dtype=save_dtype
    )


def _upload_dir_to_gcs(local_dir: str, gs_dir: str) -> None:
  """Uploads every file in ``local_dir`` to ``gs_dir`` via ``gcsfs``.

  Uses ``gcsfs`` rather than the ``gsutil``/``gcloud`` CLIs because only the
  locked venv is guaranteed on an iris worker -- the CLIs are not on the worker
  image. The export dir is flat (no subdirs), so we upload file-by-file and then
  verify the two load-critical objects landed.

  Args:
    local_dir: a local directory of already-written files.
    gs_dir: destination ``gs://bucket/prefix`` (no trailing slash).

  Raises:
    RuntimeError: if the upload does not produce ``model.safetensors`` and
      ``config.json`` at ``gs_dir``.
  """
  import gcsfs

  fs = gcsfs.GCSFileSystem()
  names = sorted(os.listdir(local_dir))
  for name in names:
    fs.put_file(os.path.join(local_dir, name), f"{gs_dir}/{name}")
  remote = {p.rsplit("/", 1)[-1] for p in fs.ls(gs_dir)}
  missing = {"model.safetensors", "config.json"} - remote
  if missing:
    raise RuntimeError(
        f"GCS upload to {gs_dir!r} incomplete; missing {sorted(missing)} "
        f"(uploaded {sorted(names)}, remote has {sorted(remote)})."
    )
