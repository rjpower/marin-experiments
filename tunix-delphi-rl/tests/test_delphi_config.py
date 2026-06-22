"""Offline tests for the config-driven Delphi loader (no weights / TPU needed).

Covers :func:`models.delphi_qwen3.delphi_config_from_hf`: that it reads the
per-size dims from ``config.json`` for both the 447M (single-file) and the ~2B
(multi-shard) Delphi points, that it reproduces the hardcoded 447M reference
config, and that it refuses a checkpoint whose rope params differ from what the
size-independent monkeypatch bakes in.
"""

import json

import jax.numpy as jnp
import pytest

from models.delphi_qwen3 import delphi_config, delphi_config_from_hf

# rope setup shared by EVERY Delphi size (the monkeypatch bakes this in).
_DELPHI_ROPE = {
    "rope_theta": 500000,
    "rope_scaling": {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    },
}

# config.json dims (verbatim from HF) for the two points we care about.
CONFIG_447M = {
    "architectures": ["Qwen3ForCausalLM"],
    "num_hidden_layers": 11,
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 128256,
    "rms_norm_eps": 1e-5,
    "tie_word_embeddings": False,
    **_DELPHI_ROPE,
}

CONFIG_1_9B = {
    "architectures": ["Qwen3ForCausalLM"],
    "num_hidden_layers": 21,
    "hidden_size": 2048,
    "intermediate_size": 8192,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "head_dim": 128,
    "vocab_size": 128256,
    "rms_norm_eps": 1e-5,
    "tie_word_embeddings": False,
    **_DELPHI_ROPE,
}


def _write_config(tmp_path, cfg: dict) -> str:
  (tmp_path / "config.json").write_text(json.dumps(cfg))
  return str(tmp_path)


def test_447m_from_hf_matches_hardcoded_reference(tmp_path):
  """The 447M read from config.json must equal the explicit delphi_config()."""
  ref = delphi_config(dtype=jnp.float32, param_dtype=jnp.float32)
  got = delphi_config_from_hf(
      _write_config(tmp_path, CONFIG_447M),
      dtype=jnp.float32,
      param_dtype=jnp.float32,
  )
  for field in (
      "num_layers",
      "vocab_size",
      "embed_dim",
      "hidden_dim",
      "num_heads",
      "head_dim",
      "num_kv_heads",
      "rope_theta",
      "norm_eps",
      "use_tied_embedding",
  ):
    assert getattr(got, field) == getattr(ref, field), field


def test_1_9b_dims_parsed(tmp_path):
  """The ~2B point parses its (wider, deeper, full-MHA) dims."""
  cfg = delphi_config_from_hf(_write_config(tmp_path, CONFIG_1_9B))
  assert cfg.num_layers == 21
  assert cfg.embed_dim == 2048
  assert cfg.hidden_dim == 8192
  assert cfg.num_heads == 16
  assert cfg.num_kv_heads == 16  # no GQA
  assert cfg.head_dim == 128
  assert cfg.rope_theta == 500000
  assert cfg.use_tied_embedding is False


def test_head_dim_falls_back_to_hidden_over_heads(tmp_path):
  cfg = dict(CONFIG_1_9B)
  del cfg["head_dim"]
  parsed = delphi_config_from_hf(_write_config(tmp_path, cfg))
  assert parsed.head_dim == 2048 // 16


def test_rejects_mismatched_rope_theta(tmp_path):
  cfg = dict(CONFIG_1_9B, rope_theta=1_000_000)
  with pytest.raises(ValueError, match="rope_theta"):
    delphi_config_from_hf(_write_config(tmp_path, cfg))


def test_rejects_mismatched_rope_scaling(tmp_path):
  cfg = dict(CONFIG_1_9B, rope_scaling={**_DELPHI_ROPE["rope_scaling"], "factor": 4.0})
  with pytest.raises(ValueError, match="rope_scaling"):
    delphi_config_from_hf(_write_config(tmp_path, cfg))


def test_rejects_non_qwen3_arch(tmp_path):
  cfg = dict(CONFIG_1_9B, architectures=["LlamaForCausalLM"])
  with pytest.raises(ValueError, match="Qwen3ForCausalLM"):
    delphi_config_from_hf(_write_config(tmp_path, cfg))
