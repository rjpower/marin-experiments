"""M5 validation: stage-2/3 arithmetic env + reward + Delphi RAW baseline.

Extends the M4 arithmetic validation to the new curriculum stages (2 multi-step
expressions, 3 linear algebra). Loads Delphi (bf16) on CPU, then for each stage:

  * asserts the reward fns score constructed right/wrong completions correctly
    (incl. NEGATIVE answers for stage 3),
  * samples a few completions with the tunix VANILLA ``Sampler`` (temp~0.8) and
    prints completion + parsed answer + reward,
  * reports Delphi's RAW pretrained greedy solve-rate over ~50 problems.

Run::

    JAX_PLATFORMS=cpu .venv/bin/python _validate_m5_arithmetic.py
"""

import random

import jax.numpy as jnp
import numpy as np
from tunix.generate import sampler as sampler_lib

from arithmetic import (
    _make_problem,
    _parse_answer,
    answer_reward,
    build_arithmetic_dataset,
    format_reward,
    metric_fn,
)
from delphi_qwen3 import DELPHI_EOS_ID, load_delphi, load_tokenizer

DELPHI_DIR = "/home/power/code/_tunix_lab/delphi"

# Stage 3 algebra prompts tokenize to ~98 tokens (4 worked equations + the query
# echoed in full), so the prompt budget must exceed that. The sampler hard-errors
# if ``max_prompt_length + max_generation_steps > cache_size``, so size the cache
# from BOTH. (M4 stages 0-1 are ~50 tokens; stage 2 ~80.)
MAX_PROMPT_LENGTH = 128
MAX_GEN_STEPS = 16  # answers are short; a few tokens suffice past the answer


def _build_sampler(model, tokenizer) -> sampler_lib.Sampler:
  """Builds a tunix vanilla Sampler for Delphi (the GRPO rollout backend)."""
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


def _reward_unit_check_stage2() -> None:
  """Asserts reward fns score stage-2 known-good/known-bad completions."""
  prompts = ["Q: 3 * 4 + 2 = A:", "Q: 5 + 2 * 6 = A:", "Q: (4 + 3) * 2 = A:"]
  gold = ["14", "17", "14"]
  good = [" 14\nQ: 1 + 1", " 17 done", " 14"]
  bad = [" 15\nQ: 1 + 1", " 18 done", " no answer"]
  assert answer_reward(prompts, good, gold) == [1.0, 1.0, 1.0]
  assert answer_reward(prompts, bad, gold) == [0.0, 0.0, 0.0]
  assert format_reward(prompts, good, gold) == [0.1, 0.1, 0.1]
  assert format_reward(prompts, bad, gold) == [0.1, 0.1, 0.0]
  print("[reward-unit stage2] PASS (good=[1,1,1] bad=[0,0,0])")


def _reward_unit_check_stage3() -> None:
  """Asserts reward fns score stage-3 completions incl. NEGATIVE answers."""
  prompts = ["...; x =", "...; x =", "...; x ="]
  gold = ["-2", "4", "-4"]  # two negative golds
  # Known-good: model emits the (possibly negative) integer first.
  good = [" -2\nSolve for x", " 4; x = 9", " -4 done"]
  # Known-bad: wrong sign or magnitude.
  bad = [" 2\nSolve", " 5; x = 9", " 4 done"]
  ar_good = answer_reward(prompts, good, gold)
  ar_bad = answer_reward(prompts, bad, gold)
  print("[reward-unit stage3] answer_reward(known-good):", ar_good)
  print("[reward-unit stage3] answer_reward(known-bad) :", ar_bad)
  assert ar_good == [1.0, 1.0, 1.0], "stage3 negative known-good must score 1.0"
  assert ar_bad == [0.0, 0.0, 0.0], "stage3 wrong-sign known-bad must score 0.0"
  # The first signed integer is the answer (a leading '-' is part of it).
  assert _parse_answer(" -2\nSolve for x") == -2
  assert _parse_answer(" -4 done") == -4
  assert format_reward(prompts, good, gold) == [0.1, 0.1, 0.1]
  print("[reward-unit stage3] PASS (negative-answer parsing correct)")


def _validate_stage(sampler, stage: int, *, n_baseline: int = 50) -> None:
  """Samples + reports baseline solve-rate for one stage."""
  print(f"\n================ STAGE {stage} ================")

  # ---- a few sampled completions at temp 0.8 (what GRPO rollout sees) -------
  rng = random.Random(0)
  problems = [_make_problem(stage, rng) for _ in range(8)]
  sample_prompts = [p for p, _ in problems]
  sample_gold = [g for _, g in problems]
  texts = _sample(sampler, sample_prompts, temperature=0.8, seed=0)
  rewards = answer_reward(sample_prompts, texts, sample_gold)
  print(f"[stage {stage} samples @ temp=0.8] query -> completion | parsed | gold | reward")
  for (p, g), text, r in list(zip(problems, texts, rewards))[:3]:
    query = p.splitlines()[-1]
    parsed = _parse_answer(str(text))
    print(f"  {query!r} -> {text!r:34} | parsed={parsed} | gold={g} | r={r}")

  # ---- RAW greedy baseline solve-rate over n_baseline problems -------------
  rng2 = random.Random(123)
  base = [_make_problem(stage, rng2) for _ in range(n_baseline)]
  base_prompts = [p for p, _ in base]
  base_gold = [g for _, g in base]
  base_texts = _sample(sampler, base_prompts, temperature=0.0, seed=0)
  base_rewards = answer_reward(base_prompts, base_texts, base_gold)
  m = metric_fn(base_prompts, base_texts, base_rewards, base_rewards)
  solve_rate = float(np.mean(base_rewards))
  print(
      f"[BASELINE stage-{stage}, greedy, n={n_baseline}] raw solve-rate = "
      f"{solve_rate:.3f} ({int(sum(base_rewards))}/{n_baseline}) "
      f"metric_solve_ratio={m['arithmetic/solve_ratio'][0]:.3f}"
  )

  # ---- dataset wiring sanity for this stage --------------------------------
  ds = build_arithmetic_dataset(stage=stage, n=16, seed=0, batch_size=8)
  row = ds[0]
  assert row["prompts"].shape == (8,) and row["answer"].shape == (8,)
  print(
      f"[dataset stage-{stage}] batch prompts{row['prompts'].shape} "
      f"answer{row['answer'].shape}; first gold={str(row['answer'][0])!r}"
  )


def main() -> None:
  tokenizer = load_tokenizer(DELPHI_DIR)
  model = load_delphi(DELPHI_DIR, dtype=jnp.bfloat16)
  sampler = _build_sampler(model, tokenizer)

  _reward_unit_check_stage2()
  _reward_unit_check_stage3()

  _validate_stage(sampler, 2)
  _validate_stage(sampler, 3)

  print("\nM5 ARITHMETIC ENV VALIDATION (stages 2,3): PASS")


if __name__ == "__main__":
  main()
