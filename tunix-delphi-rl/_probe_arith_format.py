"""Probe candidate arithmetic prompt formats by sampling Delphi greedily.

Delphi is a 447M BASE LM (no chat template). We compare a few raw-text prompt
formats on a handful of single-digit additions to see which the base model is
most likely to CONTINUE with the answer in a parseable place. Prints the greedy
continuation for each format so we can pick the format + parser.

Run::

    JAX_PLATFORMS=cpu .venv/bin/python _probe_arith_format.py
"""

import jax.numpy as jnp

from delphi_qwen3 import DELPHI_EOS_ID, load_delphi, load_tokenizer

DELPHI_DIR = "/home/power/code/_tunix_lab/delphi"


def greedy(model, tokenizer, prompt: str, *, max_new: int = 12) -> str:
  """Greedy-decodes a continuation of ``prompt`` using the KV cache."""
  ids = tokenizer.encode(prompt)
  prompt_len = len(ids)
  cache_size = prompt_len + max_new + 2
  cache = model.init_cache(
      batch_size=1, cache_size=cache_size, dtype=model.config.dtype
  )
  toks = jnp.asarray(ids, dtype=jnp.int32)[None, :]
  positions = jnp.arange(prompt_len)[None, :]
  causal = jnp.tril(jnp.ones((prompt_len, prompt_len), dtype=jnp.bool_))
  prefill_mask = jnp.zeros((1, prompt_len, cache_size), dtype=jnp.bool_)
  prefill_mask = prefill_mask.at[:, :, :prompt_len].set(causal[None])
  logits, cache = model(toks, positions, cache, prefill_mask)

  out = []
  next_tok = int(jnp.argmax(logits[0, -1]))
  out.append(next_tok)
  cur = prompt_len
  for _ in range(max_new - 1):
    if next_tok == DELPHI_EOS_ID:
      break
    step = jnp.asarray([[next_tok]], dtype=jnp.int32)
    spos = jnp.asarray([[cur]], dtype=jnp.int32)
    smask = (jnp.arange(cache_size) <= cur)[None, None, :]
    logits, cache = model(step, spos, cache, smask)
    next_tok = int(jnp.argmax(logits[0, -1]))
    out.append(next_tok)
    cur += 1
  return tokenizer.decode(out)


# Few-shot raw-text format (base LMs follow demonstrations best). The model
# should continue after the trailing "= " with the numeric answer.
FEWSHOT = (
    "Q: 2 + 3 = A: 5\n"
    "Q: 7 + 1 = A: 8\n"
    "Q: 4 + 4 = A: 8\n"
    "Q: {a} + {b} = A:"
)

# Zero-shot bare-equation format.
BARE = "{a} + {b} = "

# Zero-shot <answer> tag format.
ANSWER_TAG = (
    "Compute the sum. Put the result in <answer></answer>.\n"
    "{a} + {b} = <answer>"
)


def main() -> None:
  tokenizer = load_tokenizer(DELPHI_DIR)
  model = load_delphi(DELPHI_DIR, dtype=jnp.bfloat16)
  cases = [(3, 4), (1, 2), (6, 2), (5, 5), (8, 1)]
  for name, tmpl in [("FEWSHOT", FEWSHOT), ("BARE", BARE), ("ANSWER_TAG", ANSWER_TAG)]:
    print(f"\n===== format: {name} =====")
    for a, b in cases:
      prompt = tmpl.format(a=a, b=b)
      cont = greedy(model, tokenizer, prompt, max_new=10)
      print(f"  {a}+{b} (gold {a+b}) -> {cont!r}")


if __name__ == "__main__":
  main()
