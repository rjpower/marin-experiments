# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris smoke test for the next model class: load a stock Qwen3 base model into
tunix and confirm it loads correctly and generates coherent text on TPU.

This is the "bump to the next model class" checkpoint. It exercises the new
:mod:`models.qwen3_loader` (general GQA + tied-embedding Qwen3 loader, no Delphi
RoPE monkeypatch) end-to-end through the same tunix ``Sampler`` stack the
training code uses. Correctness evidence is twofold:
  * the loader's hard key-coverage + all-params-concrete assertions pass, and
  * the model produces sensible completions (coherent English, few-shot
    arithmetic, code completion) -- which would be garbage if RoPE / the key-map
    / GQA / tied-embeddings were wrong.

Submit (from repo root):

    uv run iris --cluster=marin job run --no-wait \
      --tpu v6e-4 --enable-extra-resources --extra tpu --region europe-west4 \
      --cpu 8 --memory 64GB --disk 60GB --max-retries 1 --job-name qwen3-smoke \
      -e HF_TOKEN "$HF_TOKEN" -- python launch_qwen3_smoke.py

Config via env: ``QWEN3_REPO`` (default ``Qwen/Qwen3-1.7B-Base``),
``QWEN3_MODEL_DIR`` (default ``./qwen3``), ``QWEN3_MAX_NEW`` (default 24).
"""

import os

import jax
import jax.numpy as jnp
from huggingface_hub import snapshot_download
from tunix.generate import sampler as sampler_lib

from models.qwen3_loader import (
    load_qwen3,
    load_qwen3_tokenizer,
    qwen3_config_from_hf,
)
from training.train_multiturn import _build_mesh

PROMPTS = [
    "The capital of France is",
    "Once upon a time, there was a small robot who",
    "3 + 4 = 7\n10 + 5 = 15\n8 + 6 = ",
    "Q: What is 12 times 3?\nA:",
    "def add(a, b):\n    return",
    "The first five prime numbers are 2, 3, 5,",
]


def _ensure(repo: str, model_dir: str) -> str:
  if not os.path.exists(os.path.join(model_dir, "config.json")):
    snapshot_download(repo_id=repo, local_dir=model_dir)
  return model_dir


def main() -> None:
  repo = os.environ.get("QWEN3_REPO", "Qwen/Qwen3-1.7B-Base")
  model_dir = os.environ.get("QWEN3_MODEL_DIR", "./qwen3")
  max_new = int(os.environ.get("QWEN3_MAX_NEW", "24"))

  print(f"[qwen3-smoke] jax {jax.__version__} devices={jax.devices()}", flush=True)
  print(f"[qwen3-smoke] repo={repo} -> {model_dir}", flush=True)
  _ensure(repo, model_dir)

  cfg = qwen3_config_from_hf(model_dir)
  print(
      f"[qwen3-smoke] config: layers={cfg.num_layers} embed={cfg.embed_dim} "
      f"hidden={cfg.hidden_dim} heads={cfg.num_heads} kv_heads={cfg.num_kv_heads} "
      f"head_dim={cfg.head_dim} vocab={cfg.vocab_size} rope_theta={cfg.rope_theta} "
      f"norm_eps={cfg.norm_eps} tied={cfg.use_tied_embedding}",
      flush=True,
  )

  mesh = _build_mesh()
  tokenizer = load_qwen3_tokenizer(model_dir)
  print(f"[qwen3-smoke] eos_id={tokenizer.eos_token_id} pad_id={tokenizer.pad_token_id}", flush=True)

  model = load_qwen3(model_dir, mesh=mesh, dtype=jnp.bfloat16)
  print("[qwen3-smoke] LOAD OK (key-coverage + all-params-concrete assertions passed)", flush=True)

  cache_size = 128 + max_new + 16
  cache_config = sampler_lib.CacheConfig(
      cache_size=cache_size,
      num_layers=model.config.num_layers,
      num_kv_heads=model.config.num_kv_heads,
      head_dim=model.config.head_dim,
  )
  sampler = sampler_lib.Sampler(transformer=model, tokenizer=tokenizer, cache_config=cache_config)

  with mesh:
    out = sampler(
        input_strings=PROMPTS,
        max_generation_steps=max_new,
        max_prompt_length=cache_size - max_new - 4,
        echo=False,
        eos_tokens=[tokenizer.eos_token_id],
        temperature=0.0,
        seed=0,
    )

  print("[qwen3-smoke] ===== GREEDY COMPLETIONS =====", flush=True)
  for prompt, text in zip(PROMPTS, out.text):
    one_line = text.replace("\n", "\\n")
    print(f"[qwen3-smoke]  {prompt!r}  ->  {one_line!r}", flush=True)
  print("[qwen3-smoke] SMOKE COMPLETE", flush=True)


if __name__ == "__main__":
  main()
