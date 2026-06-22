# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU logit-parity roundtrip for :func:`serving.export_hf.save_qwen3_to_hf`.

Constructs a small fresh-init tunix nnx Qwen3, runs a forward pass to get
reference logits, exports it to a HF safetensors directory, reloads it with the
repo's native :func:`models.qwen3_loader.load_qwen3`, runs the same forward, and
asserts the logits match (argmax identical AND ``jnp.allclose``). Covers both the
untied (has ``lm_head``) and tied-embedding paths.

Fast and CPU-only -- NOT marked ``slow``, so it runs by default. The export
tolerates an ``hf_config_dir`` that has ``config.json`` but no tokenizer files.
"""

from __future__ import annotations

import json
import os

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from tunix.models.qwen3 import model as qm

from models.qwen3_loader import load_qwen3
from serving.export_hf import save_qwen3_to_hf

# Small architecture used for both cases.
_NUM_LAYERS = 2
_EMBED_DIM = 64
_NUM_HEADS = 4
_NUM_KV_HEADS = 2
_HEAD_DIM = 16
_HIDDEN_DIM = 128
_VOCAB_SIZE = 256


def _small_config(*, use_tied_embedding: bool) -> qm.ModelConfig:
  return qm.ModelConfig(
      num_layers=_NUM_LAYERS,
      vocab_size=_VOCAB_SIZE,
      embed_dim=_EMBED_DIM,
      hidden_dim=_HIDDEN_DIM,
      num_heads=_NUM_HEADS,
      head_dim=_HEAD_DIM,
      num_kv_heads=_NUM_KV_HEADS,
      rope_theta=1_000_000,
      norm_eps=1e-6,
      use_tied_embedding=use_tied_embedding,
      dtype=jnp.float32,
      param_dtype=jnp.float32,
  )


def _write_hf_config(config_dir: str, *, tie_word_embeddings: bool) -> None:
  """Writes a minimal HF Qwen3 config.json matching the small model.

  No tokenizer files are written -- the exporter must tolerate their absence.
  """
  cfg = {
      "architectures": ["Qwen3ForCausalLM"],
      "model_type": "qwen3",
      "vocab_size": _VOCAB_SIZE,
      "hidden_size": _EMBED_DIM,
      "intermediate_size": _HIDDEN_DIM,
      "num_hidden_layers": _NUM_LAYERS,
      "num_attention_heads": _NUM_HEADS,
      "num_key_value_heads": _NUM_KV_HEADS,
      "head_dim": _HEAD_DIM,
      "rope_theta": 1_000_000,
      "rms_norm_eps": 1e-6,
      "tie_word_embeddings": tie_word_embeddings,
      "max_position_embeddings": 32,
  }
  with open(os.path.join(config_dir, "config.json"), "w") as f:
    json.dump(cfg, f)


def _forward_logits(model: qm.Qwen3, tokens: jnp.ndarray) -> jnp.ndarray:
  """Runs a single forward pass and returns float32 logits [B, L, V]."""
  positions = jnp.arange(tokens.shape[1], dtype=jnp.int32)[None, :]
  # Causal attention mask [B, L, L].
  l = tokens.shape[1]
  mask = jnp.tril(jnp.ones((l, l), dtype=jnp.bool_))[None, :, :]
  logits, _ = model(tokens, positions, cache=None, attention_mask=mask)
  return logits


@pytest.mark.parametrize("use_tied_embedding", [False, True])
def test_export_hf_roundtrip_logit_parity(tmp_path, use_tied_embedding):
  config = _small_config(use_tied_embedding=use_tied_embedding)
  model = qm.Qwen3(config, rngs=nnx.Rngs(params=0))

  tokens = jnp.array([[1, 5, 9, 13, 42, 7, 200, 3]], dtype=jnp.int32)
  ref_logits = _forward_logits(model, tokens)

  # hf_config_dir: config.json only, no tokenizer files (export must tolerate).
  hf_config_dir = tmp_path / "base_config"
  hf_config_dir.mkdir()
  _write_hf_config(str(hf_config_dir), tie_word_embeddings=use_tied_embedding)

  out_dir = tmp_path / "exported"
  save_qwen3_to_hf(
      model, str(out_dir), hf_config_dir=str(hf_config_dir), save_dtype="float32"
  )

  # Untied -> lm_head.weight present; tied -> absent.
  import safetensors.numpy as safe_np

  state = safe_np.load_file(str(out_dir / "model.safetensors"))
  if use_tied_embedding:
    assert "lm_head.weight" not in state
  else:
    assert "lm_head.weight" in state

  reloaded = load_qwen3(str(out_dir), dtype=jnp.float32)
  new_logits = _forward_logits(reloaded, tokens)

  assert jnp.array_equal(jnp.argmax(ref_logits, -1), jnp.argmax(new_logits, -1))
  assert jnp.allclose(ref_logits, new_logits, rtol=1e-4, atol=1e-4), (
      f"max abs diff = {float(jnp.max(jnp.abs(ref_logits - new_logits)))}"
  )
