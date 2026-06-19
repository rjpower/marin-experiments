"""M4 validation: HF parity of the rope MONKEYPATCH against STOCK tunix.

Re-runs M1's parity gate, but against stock tunix 0.1.7 + the import-time
monkeypatch (delphi_patch.patch_tunix_rope_for_delphi), proving the
worker-shippable rope fix delivers the SAME exact-parity fix as M1's clone edit.

Compares fp32 tunix logits against transformers' Delphi on a ~256-token prompt:
  * top-1 next-token agreement (target 100%)
  * per-logit fp32 MSE (target < 1e-3)
Also runs a no-patch control (stock apply_rope, theta=1e6, no scaling) to show
the patch is load-bearing.

Run::

    JAX_PLATFORMS=cpu .venv/bin/python _validate_m4_rope.py
"""

import jax.numpy as jnp
import numpy as np

# Capture the STOCK apply_rope BEFORE any patch import touches it, so the control
# run uses genuine stock tunix rope.
from tunix.models.qwen3 import model as qm

_STOCK_APPLY_ROPE = qm.apply_rope

import delphi_patch
from delphi_qwen3 import load_delphi, load_tokenizer
from tunix.models.qwen3 import params as qp


DELPHI_DIR = "/home/power/code/_tunix_lab/delphi"

PROMPT_TEXT = (
    "The history of science is the study of the development of science and "
    "scientific knowledge, including both the natural and social sciences. "
    "Science is a body of empirical, theoretical, and practical knowledge "
    "about the natural world, produced by scientists who emphasize the "
    "observation, explanation, and prediction of real-world phenomena. "
    "Historiography of science, in contrast, studies the methods employed by "
    "historians of science. The English word scientist is relatively recent, "
    "first coined by the polymath William Whewell in the nineteenth century."
)


def _full_logits(model: qm.Qwen3, ids_np: np.ndarray) -> np.ndarray:
  """Cache-free full-sequence fp32 logits for [T] token ids."""
  t = ids_np.shape[0]
  toks = jnp.asarray(ids_np)[None, :]
  positions = jnp.arange(t)[None, :]
  mask = jnp.tril(jnp.ones((t, t), dtype=jnp.bool_))[None, ...]
  logits, _ = model(toks, positions, None, mask)
  return np.asarray(logits[0], dtype=np.float32)


def _stock_unpatched_model() -> qm.Qwen3:
  """Loads Delphi with the STOCK (unpatched) apply_rope as a control.

  Temporarily restores the stock ``apply_rope`` (theta=1e6 default, no Llama-3
  scaling) on the tunix model module, loads Delphi against it, then leaves the
  module attribute as it found it for the caller to re-patch.
  """
  qm.apply_rope = _STOCK_APPLY_ROPE
  config = qm.ModelConfig(
      num_layers=11,
      vocab_size=128256,
      embed_dim=1024,
      hidden_dim=4096,
      num_heads=8,
      head_dim=128,
      num_kv_heads=8,
      rope_theta=500000,
      norm_eps=1e-5,
      use_tied_embedding=False,
      dtype=jnp.float32,
      param_dtype=jnp.float32,
  )
  return qp.create_model_from_safe_tensors(
      file_dir=DELPHI_DIR, config=config, dtype=jnp.float32
  )


def main() -> None:
  tokenizer = load_tokenizer(DELPHI_DIR)
  ids = tokenizer.encode(PROMPT_TEXT)[:256]
  ids_np = np.asarray(ids, dtype=np.int64)
  print(f"prompt length: {len(ids_np)} tokens")

  # HF oracle (torch).
  import torch
  from transformers import AutoModelForCausalLM

  hf = AutoModelForCausalLM.from_pretrained(DELPHI_DIR, dtype=torch.float32)
  hf.eval()
  with torch.no_grad():
    hf_logits = hf(torch.tensor(ids_np)[None, :]).logits[0].float().numpy()

  # Control: STOCK unpatched tunix (theta=1e6 default, no scaling). Forward runs
  # while qm.apply_rope is the stock fn, since Attention.block resolves the name
  # at call time.
  ctrl_model = _stock_unpatched_model()
  is_stock = qm.apply_rope is _STOCK_APPLY_ROPE
  ctrl_logits = _full_logits(ctrl_model, ids_np)
  top1_ctrl = float(
      np.mean(np.argmax(hf_logits, -1) == np.argmax(ctrl_logits, -1))
  )
  mse_ctrl = float(np.mean((hf_logits - ctrl_logits) ** 2))

  # Patched: stock tunix + monkeypatch (load_delphi calls the patch). Forward
  # runs while qm.apply_rope is the Delphi replacement.
  model = load_delphi(DELPHI_DIR, dtype=jnp.float32)
  patched = qm.apply_rope is delphi_patch._delphi_apply_rope
  tx_logits = _full_logits(model, ids_np)
  top1 = float(np.mean(np.argmax(hf_logits, -1) == np.argmax(tx_logits, -1)))
  mse = float(np.mean((hf_logits - tx_logits) ** 2))

  print(f"control apply_rope is stock tunix fn: {is_stock}")
  print(
      f"[CONTROL stock-unpatched] top-1={top1_ctrl:.4f}  fp32 MSE={mse_ctrl:.3e}"
  )
  print(f"patched apply_rope is _delphi_apply_rope: {patched}")
  print(f"[PATCHED stock+monkeypatch] top-1={top1:.4f}  fp32 MSE={mse:.3e}")

  assert patched, "monkeypatch did not take effect"
  assert top1 == 1.0, f"top-1 {top1} != 1.0 with the monkeypatch"
  assert mse < 1e-3, f"MSE {mse:.3e} >= 1e-3 with the monkeypatch"
  print("\nM4 ROPE MONKEYPATCH HF-PARITY: PASS (top-1=100%, MSE<1e-3 vs stock+patch)")


if __name__ == "__main__":
  main()
