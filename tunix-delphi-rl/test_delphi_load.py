"""M1 gate test: load Delphi into tunix Qwen3 and verify load + numerics.

Gates:
  1. KEY COVERAGE -- every safetensors tensor maps to a model param and every
     model param is concrete (no eval_shape sentinel).
  2. COHERENCE   -- greedy continuation from natural-language prompts produces
     real English words, not garbage.
  3. PERPLEXITY (torch-free) -- mean next-token cross-entropy on a fixed English
     paragraph with the CORRECT rope (theta=500000 + Llama-3 scaling) vs the
     unmodeled-scaling variants. NOTE: investigation showed the impactful bug is
     the UNMODELED Llama-3 rope SCALING, not rope_theta alone; at short context
     the perplexity differences between rope variants are tiny (~0.01 nats) and
     non-monotonic, so this gate is a coherence/sanity check, and the
     authoritative numeric gate is HF parity (gate 4).
  4. HF PARITY (authoritative) -- top-1 next-token agreement and per-logit fp32
     MSE vs transformers' Delphi. Skipped only if torch is unavailable.

Run with::

    JAX_PLATFORMS=cpu python test_delphi_load.py

or under pytest. Download of Delphi happens automatically if absent.
"""

import contextlib
import os

import jax
import jax.numpy as jnp
import numpy as np
from huggingface_hub import snapshot_download

from delphi_qwen3 import (
    DELPHI_EOS_ID,
    delphi_config,
    load_delphi,
    load_tokenizer,
)
from tunix.models.qwen3 import model as qm


DELPHI_REPO = "marin-community/delphi-3e18-447Mparams-1.2Btokens"
DELPHI_DIR = "/home/power/code/_tunix_lab/delphi"

# A fixed held-out English paragraph for the perplexity gate. Clean prose that a
# 447M base LM trained on web text should model well.
PERPLEXITY_TEXT = (
    "The history of science is the study of the development of science and "
    "scientific knowledge, including both the natural and social sciences. "
    "Science is a body of empirical, theoretical, and practical knowledge "
    "about the natural world, produced by scientists who emphasize the "
    "observation, explanation, and prediction of real-world phenomena. "
    "Historiography of science, in contrast, studies the methods employed by "
    "historians of science. The English word scientist is relatively recent, "
    "first coined by the polymath William Whewell in the nineteenth century. "
    "Before that, people investigating nature called themselves natural "
    "philosophers. While empirical investigations of the natural world have "
    "been described since antiquity, the modern scientific method took shape "
    "during the seventeenth century, in what is now known as the scientific "
    "revolution."
)

COHERENCE_PROMPTS = [
    "The capital of France is",
    "Once upon a time",
    "Water is made of hydrogen and",
]


def ensure_delphi() -> str:
  """Downloads Delphi if not already present and returns its directory."""
  if not os.path.exists(os.path.join(DELPHI_DIR, "model.safetensors")):
    snapshot_download(repo_id=DELPHI_REPO, local_dir=DELPHI_DIR)
  return DELPHI_DIR


def _causal_mask(seq_len: int) -> jax.Array:
  """Returns a [1, T, T] lower-triangular boolean causal mask."""
  return jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))[None, ...]


def full_sequence_logits(model: qm.Qwen3, token_ids: jax.Array) -> jax.Array:
  """Runs a full-sequence (cache-free) forward pass and returns fp32 logits.

  Args:
    model: a Qwen3 model.
    token_ids: integer token ids, shape [B, T].

  Returns:
    Logits, shape [B, T, V], fp32.
  """
  b, t = token_ids.shape
  positions = jnp.broadcast_to(jnp.arange(t), (b, t))
  mask = jnp.broadcast_to(_causal_mask(t), (b, t, t))
  logits, _ = model(token_ids, positions, None, mask)
  return logits


def cross_entropy(model: qm.Qwen3, token_ids: list[int]) -> float:
  """Mean next-token cross-entropy (nats) of a token sequence under the model.

  Args:
    model: a Qwen3 model.
    token_ids: a 1-D list of token ids.

  Returns:
    Mean per-token cross-entropy in nats over positions 1..T-1.
  """
  ids = jnp.asarray(token_ids, dtype=jnp.int32)[None, :]
  logits = full_sequence_logits(model, ids)[0]  # [T, V]
  log_probs = jax.nn.log_softmax(logits[:-1].astype(jnp.float32), axis=-1)
  targets = ids[0, 1:]
  tok_lp = jnp.take_along_axis(log_probs, targets[:, None], axis=-1)[:, 0]
  return float(-jnp.mean(tok_lp))


def greedy_generate(
    model: qm.Qwen3,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    cache_size: int,
) -> list[int]:
  """Greedy decode using the model's native KV cache.

  Prefills the prompt, then decodes one token at a time, threading the
  ``{'k','v','end_index'}`` ring-buffer cache. Stops at eos.

  Args:
    model: a Qwen3 model.
    prompt_ids: the prompt token ids.
    max_new_tokens: maximum number of tokens to generate.
    cache_size: KV cache length (>= len(prompt_ids) + max_new_tokens).

  Returns:
    The list of newly generated token ids (prompt excluded).
  """
  prompt_len = len(prompt_ids)
  cache = model.init_cache(
      batch_size=1, cache_size=cache_size, dtype=model.config.dtype
  )

  # Prefill: mask is [B, prompt_len, cache_size] causal over the prefilled span.
  ids = jnp.asarray(prompt_ids, dtype=jnp.int32)[None, :]
  positions = jnp.arange(prompt_len)[None, :]
  causal = jnp.tril(jnp.ones((prompt_len, prompt_len), dtype=jnp.bool_))
  prefill_mask = jnp.zeros((1, prompt_len, cache_size), dtype=jnp.bool_)
  prefill_mask = prefill_mask.at[:, :, :prompt_len].set(causal[None])
  logits, cache = model(ids, positions, cache, prefill_mask)

  next_tok = int(jnp.argmax(logits[0, -1]))
  generated = [next_tok]

  cur_pos = prompt_len
  for _ in range(max_new_tokens - 1):
    if next_tok == DELPHI_EOS_ID:
      break
    step_ids = jnp.asarray([[next_tok]], dtype=jnp.int32)
    step_pos = jnp.asarray([[cur_pos]], dtype=jnp.int32)
    # Attend to all already-written cache slots (0..cur_pos inclusive).
    step_mask = (jnp.arange(cache_size) <= cur_pos)[None, None, :]
    logits, cache = model(step_ids, step_pos, cache, step_mask)
    next_tok = int(jnp.argmax(logits[0, -1]))
    generated.append(next_tok)
    cur_pos += 1

  return generated


def _fraction_alpha_words(text: str) -> float:
  """Returns the fraction of whitespace tokens that are alphabetic words."""
  words = text.split()
  if not words:
    return 0.0
  alpha = sum(1 for w in words if any(c.isalpha() for c in w))
  return alpha / len(words)


def test_delphi_m1_gates():
  """Runs all M1 gates and asserts the thresholds. Prints real numbers."""
  model_dir = ensure_delphi()
  tokenizer = load_tokenizer(model_dir)

  # ---- Gate 1: key coverage (load_delphi hard-asserts internally) ----------
  model = load_delphi(model_dir, dtype=jnp.bfloat16)
  # Re-run the explicit count for the report.
  from delphi_qwen3 import _assert_key_coverage  # noqa: PLC0415

  keys = _assert_key_coverage(model_dir, delphi_config())
  print(f"[GATE 1 key-coverage] PASS: {len(keys)}/{len(keys)} tensors mapped, "
        f"all params concrete.")

  # ---- Gate 2: coherence ---------------------------------------------------
  cache_size = 64
  coherent = True
  for prompt in COHERENCE_PROMPTS:
    ids = tokenizer.encode(prompt)
    gen = greedy_generate(
        model, ids, max_new_tokens=20, cache_size=cache_size
    )
    text = tokenizer.decode(gen)
    frac = _fraction_alpha_words(text)
    print(f"[GATE 2 coherence] {prompt!r} -> {text!r}  "
          f"(alpha-word frac={frac:.2f})")
    coherent = coherent and (frac >= 0.6)
  assert coherent, "Greedy continuations were not coherent English."
  print("[GATE 2 coherence] PASS")

  # ---- Gate 3: perplexity, correct rope vs unmodeled-scaling variants ------
  # Compute in fp32 so bf16 rounding does not swamp the (small) rope effect.
  token_ids = tokenizer.encode(PERPLEXITY_TEXT)
  print(f"[GATE 3 perplexity] paragraph length: {len(token_ids)} tokens")

  fp32_model = load_delphi(model_dir, dtype=jnp.float32)
  ce_correct = cross_entropy(fp32_model, token_ids)

  # The two unmodeled variants the design doc considered: plain theta=500000
  # (its proposed "fix", but WITHOUT the required Llama-3 scaling) and plain
  # theta=1_000_000 (the original bug). Both omit the Llama-3 scaling. On stock
  # tunix these are produced by temporarily installing a plain-rope apply_rope
  # (the override restores the monkeypatch on exit), since stock ModelConfig has
  # no rope_scaling field. The forward must run while the override is active.
  with _apply_rope_override(_plain_apply_rope_factory(500000)):
    ce_no_scaling = cross_entropy(fp32_model, token_ids)
  with _apply_rope_override(_plain_apply_rope_factory(1_000_000)):
    ce_theta_1m = cross_entropy(fp32_model, token_ids)

  print(f"[GATE 3 perplexity] CE theta=500000 + Llama-3 scaling (CORRECT): "
        f"{ce_correct:.4f} nats  (ppl={np.exp(ce_correct):.2f})")
  print(f"[GATE 3 perplexity] CE theta=500000, NO scaling (doc 'fix'):     "
        f"{ce_no_scaling:.4f} nats  (ppl={np.exp(ce_no_scaling):.2f})")
  print(f"[GATE 3 perplexity] CE theta=1000000, NO scaling (orig bug):     "
        f"{ce_theta_1m:.4f} nats  (ppl={np.exp(ce_theta_1m):.2f})")
  # Sanity bound: a correctly-loaded 447M base LM on clean English.
  assert ce_correct < 3.5, (
      f"Correct-rope CE {ce_correct:.4f} exceeds the 3.5-nat sanity bound."
  )
  print("[GATE 3 perplexity] PASS (correct-rope CE < 3.5 nats; see HF parity "
        "for the authoritative rope check)")

  # ---- Gate 4: HF parity (authoritative) -----------------------------------
  parity = _hf_parity(model_dir, tokenizer)
  if parity is None:
    print("[GATE 4 HF-parity] SKIPPED (torch unavailable) -- relying on "
          "perplexity + coherence gates only.")
  else:
    top1_correct, mse_correct, top1_noscale, mse_noscale = parity
    print(f"[GATE 4 HF-parity] CORRECT rope (theta=500000 + Llama-3): "
          f"top-1={top1_correct:.4f}  fp32 MSE={mse_correct:.3e}")
    print(f"[GATE 4 HF-parity] no-scaling rope (theta=500000):        "
          f"top-1={top1_noscale:.4f}  fp32 MSE={mse_noscale:.3e}")
    assert top1_correct == 1.0, (
        f"HF top-1 agreement {top1_correct:.4f} != 1.0; rope/scaling wrong."
    )
    assert mse_correct < 1e-3, (
        f"HF per-logit MSE {mse_correct:.3e} >= 1e-3; rope/scaling wrong."
    )
    print("[GATE 4 HF-parity] PASS (top-1 == 100%, MSE < 1e-3)")

  print("\nM1 GATES PASSED.")


def _plain_apply_rope_factory(rope_theta):
  """Builds a stock-style ``apply_rope`` (no Llama-3 scaling) at a fixed theta.

  Used for the rope ablations on STOCK tunix, where ``ModelConfig`` has no
  ``rope_scaling`` field and the call sites never pass ``rope_theta`` -- so a
  baked-in replacement is the only way to install a specific theta-without-
  scaling rope. Mirrors stock ``apply_rope`` math exactly.
  """

  def _apply_rope(inputs, positions, head_dim, rope_theta=rope_theta):
    fraction = 2 * jnp.arange(0, head_dim // 2, dtype=jnp.float32) / head_dim
    timescale = rope_theta**fraction
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

  return _apply_rope


@contextlib.contextmanager
def _apply_rope_override(fn):
  """Temporarily installs ``fn`` as ``qm.apply_rope`` for the block's forwards.

  ``Attention.block`` resolves ``apply_rope`` as a module global at call time, so
  swapping the module attribute changes the rope used by any forward run inside
  the ``with`` block. Restores the previous function on exit.
  """
  prev = qm.apply_rope
  qm.apply_rope = fn
  try:
    yield
  finally:
    qm.apply_rope = prev


def _hf_parity(model_dir, tokenizer):
  """Strict HF parity. Returns (top1_c, mse_c, top1_ns, mse_ns) or None.

  Loads Delphi via ``transformers.AutoModelForCausalLM`` (torch CPU) and
  compares fp32 logits against tunix on a fixed multi-hundred-token prompt, for
  both the correct rope (Llama-3 scaling) and the no-scaling variant. Returns
  None if torch is not installed.
  """
  try:
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM  # noqa: PLC0415
  except ImportError:
    return None

  hf = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
  hf.eval()

  prompt = PERPLEXITY_TEXT + " " + PERPLEXITY_TEXT
  ids = tokenizer.encode(prompt)[:384]
  ids_np = np.asarray(ids, dtype=np.int64)

  with torch.no_grad():
    hf_logits = hf(torch.tensor(ids_np)[None, :]).logits[0].float().numpy()

  def _compare(model):
    tx = np.asarray(
        full_sequence_logits(model, jnp.asarray(ids_np)[None, :])[0]
    )
    top1 = float(np.mean(np.argmax(hf_logits, -1) == np.argmax(tx, -1)))
    mse = float(np.mean((hf_logits - tx) ** 2))
    return top1, mse

  # Correct rope: load_delphi installs the Delphi monkeypatch.
  model = load_delphi(model_dir, dtype=jnp.float32)
  top1_c, mse_c = _compare(model)
  # No-scaling variant: same weights, plain rope at theta=500000.
  with _apply_rope_override(_plain_apply_rope_factory(500000)):
    top1_ns, mse_ns = _compare(model)
  return top1_c, mse_c, top1_ns, mse_ns


if __name__ == "__main__":
  test_delphi_m1_gates()
