"""M4 validation: arithmetic env + reward + Delphi baseline solve-rate.

Loads Delphi (bf16) on CPU, samples stage-0 completions with the tunix VANILLA
``Sampler`` (the rollout backend GRPO uses), prints completions + parsed answers
+ rewards, asserts the reward fn scores constructed right/wrong completions
correctly, and reports Delphi's RAW pretrained solve-rate over ~50 stage-0
problems.

Run::

    JAX_PLATFORMS=cpu .venv/bin/python _validate_m4_arithmetic.py
"""

import jax.numpy as jnp
import numpy as np
from tunix.generate import sampler as sampler_lib

from arithmetic import (
    _make_problem,
    answer_reward,
    build_arithmetic_dataset,
    format_reward,
    metric_fn,
)
from delphi_qwen3 import DELPHI_EOS_ID, load_delphi, load_tokenizer

DELPHI_DIR = "/home/power/code/_tunix_lab/delphi"

MAX_PROMPT_LENGTH = 64
MAX_GEN_STEPS = 12  # arithmetic answers are short; a few tokens suffice


def _build_sampler(model, tokenizer) -> sampler_lib.Sampler:
  """Builds a tunix vanilla Sampler for Delphi."""
  cache_config = sampler_lib.CacheConfig(
      cache_size=MAX_PROMPT_LENGTH + MAX_GEN_STEPS + 8,
      num_layers=model.config.num_layers,
      num_kv_heads=model.config.num_kv_heads,
      head_dim=model.config.head_dim,
  )
  return sampler_lib.Sampler(
      transformer=model, tokenizer=tokenizer, cache_config=cache_config
  )


def _sample(sampler, prompts: list[str], *, temperature: float, seed: int):
  """Samples completions for a list of prompts; returns list[str] of text."""
  out = sampler(
      input_strings=prompts,
      max_generation_steps=MAX_GEN_STEPS,
      max_prompt_length=MAX_PROMPT_LENGTH,
      echo=False,
      eos_tokens=[DELPHI_EOS_ID],
      temperature=temperature,
      seed=seed,
  )
  return out.text


def main() -> None:
  tokenizer = load_tokenizer(DELPHI_DIR)
  model = load_delphi(DELPHI_DIR, dtype=jnp.bfloat16)
  sampler = _build_sampler(model, tokenizer)

  # ---- Reward fn unit check on constructed right/wrong completions ----------
  prompts = ["Q: 3 + 4 = A:", "Q: 5 + 5 = A:", "Q: 9 + 0 = A:"]
  gold = ["7", "10", "9"]
  good = [" 7\nQ: 8 + 1", " 10 done", " 9"]
  bad = [" 8\nQ: 8 + 1", " 11 done", " no answer"]
  ar_good = answer_reward(prompts, good, gold)
  ar_bad = answer_reward(prompts, bad, gold)
  fr_good = format_reward(prompts, good, gold)
  fr_bad = format_reward(prompts, bad, gold)
  print("[reward-unit] answer_reward(known-good):", ar_good)
  print("[reward-unit] answer_reward(known-bad) :", ar_bad)
  print("[reward-unit] format_reward(known-good):", fr_good)
  print("[reward-unit] format_reward(known-bad) :", fr_bad)
  assert ar_good == [1.0, 1.0, 1.0], "answer_reward failed on known-good"
  assert ar_bad == [0.0, 0.0, 0.0], "answer_reward failed on known-bad"
  assert fr_good == [0.1, 0.1, 0.1], "format_reward failed on known-good"
  assert fr_bad == [0.1, 0.1, 0.0], "format_reward: bad[2] has no integer"
  print("[reward-unit] PASS")

  # ---- Sample 16 stage-0 completions; show parsed answers + rewards ---------
  rng = __import__("random").Random(0)
  problems = [_make_problem(0, rng) for _ in range(16)]
  sample_prompts = [p for p, _ in problems]
  sample_gold = [g for _, g in problems]
  texts = _sample(sampler, sample_prompts, temperature=0.8, seed=0)
  rewards = answer_reward(sample_prompts, texts, sample_gold)
  print("\n[sample x16 @ temp=0.8] query -> completion | parsed | gold | reward")
  from arithmetic import _parse_answer  # noqa: PLC0415

  for (p, g), text, r in list(zip(problems, texts, rewards))[:6]:
    query = p.splitlines()[-1]
    parsed = _parse_answer(str(text))
    print(f"  {query!r} -> {text!r:32} | parsed={parsed} | gold={g} | r={r}")
  print(f"  (showing 6 of 16) sampled solve count = {int(sum(rewards))}/16")

  # ---- RAW baseline solve-rate over ~50 stage-0 problems (greedy) -----------
  rng2 = __import__("random").Random(123)
  base_problems = [_make_problem(0, rng2) for _ in range(50)]
  base_prompts = [p for p, _ in base_problems]
  base_gold = [g for _, g in base_problems]
  base_texts = _sample(sampler, base_prompts, temperature=0.0, seed=0)
  base_rewards = answer_reward(base_prompts, base_texts, base_gold)
  m = metric_fn(base_prompts, base_texts, base_rewards, base_rewards)
  solve_rate = float(np.mean(base_rewards))
  print(
      f"\n[BASELINE stage-0, greedy, n=50] raw solve-rate = {solve_rate:.3f} "
      f"({int(sum(base_rewards))}/50)"
  )
  print(f"[BASELINE] metric_fn: solve_ratio={m['arithmetic/solve_ratio'][0]:.3f}")

  # ---- Dataset wiring sanity ------------------------------------------------
  ds = build_arithmetic_dataset(stage=0, n=16, seed=0, batch_size=8)
  row = ds[0]
  assert row["prompts"].shape == (8,) and row["answer"].shape == (8,)
  print(
      f"\n[dataset] batch row: prompts{row['prompts'].shape} "
      f"answer{row['answer'].shape}; first gold={str(row['answer'][0])!r}"
  )
  print("\nM4 ARITHMETIC ENV VALIDATION: PASS")


if __name__ == "__main__":
  main()
