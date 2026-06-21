# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris entrypoint for conversational chat-SFT (stage 1 of the agentic ladder).

Teaches a BASE LM (Delphi-1.9B or Qwen3-1.7B-Base) a turn-taking instruction
format by SFT on a tulu mixture (see :mod:`training.chat_sft`), then shows
before/after generations on a fixed held-out instruction set so the effect is
visible. The qualitative signal: after SFT the model produces a single bounded
assistant turn that STOPS (emits EOS) instead of the base model's rambling /
turn-leaking continuation.

Config via env:
  * ``CHAT_MODEL`` (default ``delphi-1.9b``) -- registry key; use ``qwen3-1.7b-base``
    for the control arm. ``CHAT_MODEL_DIR`` (default ``./<model name>``).
  * ``CHAT_DATASET`` (default ``allenai/tulu-3-sft-mixture``).
  * ``CHAT_SFT_STEPS`` (1000), ``CHAT_BATCH_SIZE`` (8), ``CHAT_LR`` (1e-5).
  * ``CHAT_MAX_SEQ_LEN`` (1024), ``CHAT_SEED`` (0).
  * ``CHAT_EVAL_TOKENS`` (200), ``CHAT_EVAL_TEMP`` (0.7).

Submit on a single-host TPU (the 1.9B fp32 actor + AdamW wants v6e-8):

    uv run iris --cluster=marin job run --no-wait \
      --tpu v6e-8 --enable-extra-resources --extra tpu --region europe-west4 \
      --cpu 8 --memory 200GB --disk 60GB --max-retries 1 --job-name chat-sft-delphi \
      -e HF_TOKEN "$HF_TOKEN" -e CHAT_MODEL delphi-1.9b \
      -- python launch_chat_sft.py
"""

import os

import jax
import jax.numpy as jnp
from huggingface_hub import snapshot_download
from tunix.generate import sampler as sampler_lib

from models.registry import get_model_spec
from training.chat_sft import CHAT_ROLE_HEADER, format_user_prompt, run_chat_sft
from training.train_multiturn import _build_mesh

# Fixed, diverse held-out instructions: a fact, generation, explanation, code,
# arithmetic word problem, list, translation, and a beginner summary.
EVAL_INSTRUCTIONS = [
    "What is the capital of France?",
    "Write a haiku about the ocean.",
    "Explain in one sentence why the sky is blue.",
    "Write a Python function that returns the factorial of a number n.",
    "If a train travels 60 miles in 1.5 hours, what is its average speed?",
    "List three tips for staying focused while studying.",
    "Translate 'good morning' into Spanish.",
    "Summarize what a neural network is for a complete beginner.",
]

_ROLE_MARKERS = tuple(h.strip() for h in CHAT_ROLE_HEADER.values())


def _ensure_model(repo: str, model_dir: str) -> str:
  if not os.path.exists(os.path.join(model_dir, "config.json")):
    snapshot_download(repo_id=repo, local_dir=model_dir)
  return model_dir


def _generate(sampler, mesh, prompts, *, max_new, temperature, eos_id):
  with mesh:
    out = sampler(
        input_strings=prompts,
        max_generation_steps=max_new,
        max_prompt_length=256,
        echo=False,
        eos_tokens=[eos_id],
        temperature=temperature,
        top_p=1.0,  # REQUIRED: without top_p the tunix Sampler decodes greedily
        seed=0,
    )
  return out.text


def _report(label, instructions, responses) -> None:
  """Prints each response and a turn-leak / length proxy for chat-format uptake."""
  leaked = 0
  total_chars = 0
  print(f"[chat-sft] ===== {label} generations =====", flush=True)
  for instr, resp in zip(instructions, responses):
    is_leak = any(m in resp for m in _ROLE_MARKERS)
    leaked += int(is_leak)
    total_chars += len(resp)
    one_line = resp.replace("\n", "\\n")
    if len(one_line) > 240:
      one_line = one_line[:240] + "..."
    flag = " [LEAK]" if is_leak else ""
    print(f"[chat-sft]  Q: {instr}", flush=True)
    print(f"[chat-sft]  A:{flag} {one_line!r}", flush=True)
  n = len(instructions)
  print(
      f"[chat-sft] {label} SUMMARY: turn_leak={leaked}/{n} "
      f"mean_chars={total_chars / max(n, 1):.0f}",
      flush=True,
  )


def main() -> None:
  model_name = os.environ.get("CHAT_MODEL", "delphi-1.9b")
  dataset_name = os.environ.get("CHAT_DATASET", "allenai/tulu-3-sft-mixture")
  steps = int(os.environ.get("CHAT_SFT_STEPS", "1000"))
  batch_size = int(os.environ.get("CHAT_BATCH_SIZE", "8"))
  learning_rate = float(os.environ.get("CHAT_LR", "1e-5"))
  max_seq_len = int(os.environ.get("CHAT_MAX_SEQ_LEN", "1024"))
  seed = int(os.environ.get("CHAT_SEED", "0"))
  eval_tokens = int(os.environ.get("CHAT_EVAL_TOKENS", "200"))
  eval_temp = float(os.environ.get("CHAT_EVAL_TEMP", "0.7"))

  model_spec = get_model_spec(model_name)
  model_dir = os.environ.get("CHAT_MODEL_DIR") or f"./{model_spec.name}"

  print(f"[chat-sft] jax {jax.__version__} devices={jax.devices()}", flush=True)
  print(
      f"[chat-sft] model={model_spec.name} dataset={dataset_name} steps={steps} "
      f"bs={batch_size} lr={learning_rate} max_seq_len={max_seq_len} "
      f"eval_tokens={eval_tokens} eval_temp={eval_temp}",
      flush=True,
  )

  _ensure_model(model_spec.repo, model_dir)
  print(f"[chat-sft] repo={model_spec.repo} ready at {model_dir}", flush=True)

  mesh = _build_mesh()
  tokenizer = model_spec.load_tokenizer(model_dir)
  model = model_spec.load_model(model_dir, dtype=jnp.float32, mesh=mesh)
  print("[chat-sft] LOAD OK", flush=True)

  cache_size = 256 + eval_tokens + 16
  cache_config = sampler_lib.CacheConfig(
      cache_size=cache_size,
      num_layers=model.config.num_layers,
      num_kv_heads=model.config.num_kv_heads,
      head_dim=model.config.head_dim,
  )
  prompts = [format_user_prompt(q) for q in EVAL_INSTRUCTIONS]

  # ---- before SFT (base model in the chat format) ----
  sampler = sampler_lib.Sampler(transformer=model, tokenizer=tokenizer, cache_config=cache_config)
  before = _generate(
      sampler, mesh, prompts,
      max_new=eval_tokens, temperature=eval_temp, eos_id=tokenizer.eos_token_id,
  )
  _report("BEFORE-SFT", EVAL_INSTRUCTIONS, before)

  # ---- chat-SFT ----
  model = run_chat_sft(
      model, tokenizer,
      dataset_name=dataset_name,
      steps=steps,
      batch_size=batch_size,
      learning_rate=learning_rate,
      mesh=mesh,
      max_seq_len=max_seq_len,
      seed=seed,
  )

  # ---- after SFT (rebuild the sampler on the warmed weights) ----
  sampler = sampler_lib.Sampler(transformer=model, tokenizer=tokenizer, cache_config=cache_config)
  after = _generate(
      sampler, mesh, prompts,
      max_new=eval_tokens, temperature=eval_temp, eos_id=tokenizer.eos_token_id,
  )
  _report("AFTER-SFT", EVAL_INSTRUCTIONS, after)
  print(f"[chat-sft] CHAT-SFT COMPLETE (model={model_spec.name} dataset={dataset_name} steps={steps})", flush=True)


if __name__ == "__main__":
  main()
