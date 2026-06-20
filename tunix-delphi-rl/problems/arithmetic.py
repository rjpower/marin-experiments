"""Single-turn arithmetic GRPO environment for Delphi (gsm8k-style).

Delphi is a 447M BASE LM with NO chat template, so prompts are RAW few-shot text.
Empirically (see ``_probe_arith_format.py``) the base model follows a few-shot
``Q: a + b = A: n`` demonstration format the most reliably: after the trailing
``A:`` it emits a single number, then a newline and a fresh ``Q:``. The format
is followed even when the value is wrong (the value is what GRPO improves). So:

  * prompt = a few worked few-shot examples + the query, ending exactly where the
    model should emit its number (``A:`` for stages 0-2, ``x =`` for stage 3).
  * parser = the FIRST (optionally signed) integer in the completion.

CRITICAL — Delphi is a BASE LM: it only follows the format if the few-shot
DEMONSTRATIONS match the task type. So each stage carries its OWN few-shot prefix
(:data:`_FEWSHOT_PREFIXES`) of worked examples *of that stage's task* in that
stage's exact format. A stage-0 addition prefix in front of an algebra query
makes the model continue with more additions, not the answer; the per-stage
prefix is what makes the base LM emit a parseable answer for each task type.

This module provides:
  * :func:`build_arithmetic_dataset` -- a grain dataset with ``prompts`` (raw
    strings) and ``answer`` (gold string) columns, in the M2-proven grain shape
    (``grain.MapDataset.source(...).batch(n).map(...)``) that avoids the HF
    ``datasets.batch`` list-column corruption under tunix's ``tree_map``.
  * :func:`answer_reward` / :func:`format_reward` -- GRPO reward fns with the
    ``(prompts, completions, answer, **kwargs) -> list[float]`` signature.
  * :func:`metric_fn` -- solve_ratio/solve_all/solve_none for dashboards.

Stages:
  * 0: single-digit ``a + b`` (a, b in 0..9).
  * 1: ``a OP b`` with ``OP`` in ``{+, -, *}`` and operands up to 2 digits
    (0..99); subtraction is ordered so the gold answer is non-negative.
  * 2: two-operation integer expressions with normal precedence, e.g.
    ``a * b + c``, ``a + b * c``, ``(a + b) * c``, ``a + b + c`` (small operands,
    integer answer).
  * 3: basic linear algebra ``Solve for x: a*x + b = c`` with integer a, b and c
    chosen so x is a (possibly negative) small integer. Gold answer = x. The
    prompt ends ``...; x =`` so the model emits x first.
"""

import random
import re

import grain.python as grain
import numpy as np


# Per-stage few-shot prefixes. Each is a handful of worked examples *of that
# stage's task*, in that stage's exact answer format, prepended to every query.
# A base LM imitates the demonstrated task+format, so stage 2/3 MUST demonstrate
# stage-2/3 problems (not addition) or the model continues with the wrong task.
#
# Stages 0-2 share the ``Q: <expr> = A: n`` shape (prompt ends at ``A:``); stage
# 3 uses ``Solve for x: ...; x = n`` (prompt ends at ``x =``) so the model emits
# the value of x first.
_FEWSHOT_PREFIXES: dict[int, str] = {
    0: (
        "Q: 2 + 3 = A: 5\n"
        "Q: 7 + 1 = A: 8\n"
        "Q: 4 + 4 = A: 8\n"
    ),
    1: (
        "Q: 12 + 9 = A: 21\n"
        "Q: 40 - 15 = A: 25\n"
        "Q: 7 * 6 = A: 42\n"
    ),
    2: (
        "Q: 3 * 4 + 2 = A: 14\n"
        "Q: 5 + 2 * 6 = A: 17\n"
        "Q: (4 + 3) * 2 = A: 14\n"
        "Q: 9 + 8 + 6 = A: 23\n"
    ),
    3: (
        "Solve for x: 2*x + 3 = 11; x = 4\n"
        "Solve for x: 5*x - 4 = 16; x = 4\n"
        "Solve for x: 3*x + 7 = 1; x = -2\n"
        "Solve for x: 4*x + 6 = -10; x = -4\n"
    ),
}

# Matches the first (optionally signed) integer anywhere in a string. The leading
# ``-?`` makes negative answers (stage 3) parse from a leading minus sign.
_FIRST_INT_RE = re.compile(r"-?\d+")


def _make_stage2_problem(rng: random.Random) -> tuple[str, int]:
  """Generates one stage-2 two-operation integer expression and its value.

  Picks one of four small-operand templates respecting normal precedence so the
  answer is always a unique integer:

    * ``a * b + c`` / ``a + b * c`` (multiplication binds first),
    * ``(a + b) * c`` (explicit grouping),
    * ``a + b + c`` (three-term sum).

  Args:
    rng: a seeded ``random.Random`` for reproducibility.

  Returns:
    A tuple ``(expr, value)`` where ``expr`` is the expression string (no
    trailing ``=``) and ``value`` is the integer result.
  """
  template = rng.choice(["mul_add", "add_mul", "group_mul", "add_add"])
  a = rng.randint(1, 12)
  b = rng.randint(1, 12)
  c = rng.randint(1, 12)
  if template == "mul_add":
    return f"{a} * {b} + {c}", a * b + c
  if template == "add_mul":
    return f"{a} + {b} * {c}", a + b * c
  if template == "group_mul":
    return f"({a} + {b}) * {c}", (a + b) * c
  return f"{a} + {b} + {c}", a + b + c


def _make_stage3_problem(rng: random.Random) -> tuple[str, int]:
  """Generates one stage-3 linear equation ``a*x + b = c`` with integer x.

  Picks a (possibly negative) integer solution ``x`` and an integer ``b`` first,
  then sets ``c = a*x + b`` so the equation has the exact integer solution ``x``.
  ``a`` is a small positive integer (>= 1). Roughly half the problems have a
  negative ``x`` so the negative-answer parse path is exercised.

  Args:
    rng: a seeded ``random.Random`` for reproducibility.

  Returns:
    A tuple ``(expr, x)`` where ``expr`` is the ``a*x + b = c`` equation string
    (no trailing ``; x =``) and ``x`` is the integer solution (gold answer).
  """
  a = rng.randint(1, 9)
  x = rng.randint(-9, 9)
  b = rng.randint(-9, 9)
  c = a * x + b
  # Render b with an explicit sign so the equation reads naturally, e.g.
  # "2*x - 3 = 5" rather than "2*x + -3 = 5".
  if b >= 0:
    expr = f"{a}*x + {b} = {c}"
  else:
    expr = f"{a}*x - {abs(b)} = {c}"
  return expr, x


def _make_problem(stage: int, rng: random.Random) -> tuple[str, str]:
  """Generates one ``(prompt, gold_answer)`` pair for the given stage.

  The prompt is the stage's few-shot prefix followed by the query, ending
  exactly where the model should emit its answer (``A:`` for stages 0-2, ``x =``
  for stage 3) so the first integer in the completion is the model's answer.

  Args:
    stage: curriculum stage (0 single-digit add; 1 add/sub/mul to 2 digits;
      2 two-operation expressions; 3 linear algebra ``a*x + b = c``).
    rng: a seeded ``random.Random`` for reproducibility.

  Returns:
    A tuple ``(prompt, gold)`` where ``prompt`` is the raw few-shot text and
    ``gold`` is the integer answer as a string.

  Raises:
    ValueError: if ``stage`` is not a supported stage.
  """
  prefix = _FEWSHOT_PREFIXES.get(stage)
  if prefix is None:
    raise ValueError(f"Unsupported arithmetic stage: {stage}")

  if stage == 0:
    a = rng.randint(0, 9)
    b = rng.randint(0, 9)
    expr, gold = f"{a} + {b}", a + b
    prompt = f"{prefix}Q: {expr} = A:"
  elif stage == 1:
    op = rng.choice(["+", "-", "*"])
    a = rng.randint(0, 99)
    b = rng.randint(0, 99)
    if op == "+":
      gold = a + b
    elif op == "-":
      # Order operands so the answer is non-negative (cleaner gold parse).
      a, b = max(a, b), min(a, b)
      gold = a - b
    else:
      gold = a * b
    prompt = f"{prefix}Q: {a} {op} {b} = A:"
  elif stage == 2:
    expr, gold = _make_stage2_problem(rng)
    prompt = f"{prefix}Q: {expr} = A:"
  else:  # stage == 3
    expr, gold = _make_stage3_problem(rng)
    prompt = f"{prefix}Solve for x: {expr}; x ="
  return prompt, str(gold)


class _ArithmeticSource(grain.RandomAccessDataSource):
  """A grain source of ``{'prompt', 'answer'}`` arithmetic rows.

  Problems are pre-generated deterministically from ``seed`` so every epoch /
  worker sees the same rows in the same order (grain shuffling is left to the
  learner if desired).
  """

  def __init__(self, stage: int, n: int, seed: int):
    """Builds ``n`` problems for ``stage`` using ``seed``.

    Args:
      stage: curriculum stage.
      n: number of problems.
      seed: PRNG seed.
    """
    rng = random.Random(seed)
    self._rows = [_make_problem(stage, rng) for _ in range(n)]

  def __len__(self) -> int:
    return len(self._rows)

  def __getitem__(self, idx: int) -> tuple[str, str]:
    return self._rows[idx]


def build_arithmetic_dataset(
    stage: int, n: int, seed: int, batch_size: int
) -> grain.MapDataset:
  """Builds a batched grain dataset of arithmetic problems for GRPO.

  Emits rows with a ``prompts`` column (raw few-shot strings) and an ``answer``
  column (gold strings). Uses the M2-proven grain shape so each batched column
  is a single ``numpy`` array leaf: HF ``datasets.batch`` instead yields Python
  lists per column, which tunix's ``jax.tree.map(np.repeat, ...)`` recurses into
  and corrupts. ``GRPOLearner`` forwards the ``answer`` column to reward fns as
  the ``answer=`` kwarg.

  Args:
    stage: curriculum stage (0, 1, 2, or 3).
    n: number of distinct problems to generate.
    seed: PRNG seed for the problem set.
    batch_size: prompts per global step.

  Returns:
    A batched ``grain.MapDataset`` with ``prompts`` and ``answer`` columns.
  """
  source = _ArithmeticSource(stage, n, seed)

  def _to_columns(batch):
    # grain's ``.batch`` collates a tuple-valued source row-wise PER FIELD: a
    # batch of ``(prompt, gold)`` tuples becomes a 2-tuple
    # ``(prompts_array, answers_array)`` of numpy string arrays (each a single
    # leaf, which is what tunix's tree_map needs).
    prompts, answers = batch
    return {"prompts": prompts, "answer": answers}

  return grain.MapDataset.source(source).batch(batch_size).map(_to_columns)


def _parse_answer(completion: str) -> int | None:
  """Parses the model's numeric answer: the FIRST integer in the completion.

  With the few-shot format the model emits a single number immediately after the
  prompt's answer marker (``A:`` for stages 0-2, ``x =`` for stage 3), so the
  first (optionally signed) integer in the completion is the answer. The leading
  ``-?`` in :data:`_FIRST_INT_RE` makes a leading minus sign parse, so stage-3
  negative solutions (e.g. ``" -2\n"``) are read as ``-2``.

  Args:
    completion: the decoded completion string (text after the prompt).

  Returns:
    The first integer found, or ``None`` if the completion has no integer.
  """
  match = _FIRST_INT_RE.search(completion)
  if match is None:
    return None
  return int(match.group())


def answer_reward(prompts, completions, answer, **kwargs) -> list[float]:
  """Reward 1.0 for an exact integer-match answer, else 0.0.

  Args:
    prompts: the batch of prompt strings (unused).
    completions: the batch of decoded completion strings.
    answer: the batch of gold answer strings (forwarded dataset column).
    **kwargs: other forwarded dataset columns (unused).

  Returns:
    One float per completion: 1.0 if the parsed integer equals the gold
    integer, else 0.0.
  """
  del prompts, kwargs
  rewards: list[float] = []
  for completion, gold in zip(completions, answer):
    gold = str(np.asarray(gold).item()) if not isinstance(gold, str) else gold
    parsed = _parse_answer(str(completion))
    rewards.append(1.0 if (parsed is not None and parsed == int(gold)) else 0.0)
  return rewards


def format_reward(prompts, completions, answer, **kwargs) -> list[float]:
  """Small reward (0.1) for emitting any parseable integer.

  Encourages the model to keep producing a number to parse (so ``answer_reward``
  has something to score) without overwhelming the correctness signal. Summed
  with :func:`answer_reward` by the learner.

  Args:
    prompts: the batch of prompt strings (unused).
    completions: the batch of decoded completion strings.
    answer: the batch of gold answer strings (unused).
    **kwargs: other forwarded dataset columns (unused).

  Returns:
    One float per completion: 0.1 if any integer is present, else 0.0.
  """
  del prompts, answer, kwargs
  return [
      0.1 if _parse_answer(str(completion)) is not None else 0.0
      for completion in completions
  ]


_PROXIMITY_SCALE = 10.0


def proximity_reward(prompts, completions, answer, **kwargs) -> list[float]:
  """Dense distance-shaped reward: 1.0 exact, else partial credit by closeness.

  Densifies the exact-match signal so GRPO has non-zero advantage *within a
  group* even when no sampled completion is exactly right. The harder curriculum
  stages (2-3) start near a 0% solve rate, where :func:`answer_reward` gives every
  sample the same 0.0 -> the group-relative advantage is identically zero -> no
  gradient (the difficulty-ramp wall). This reward instead scores a near-miss by
  how close it is, so "off by 1" beats "off by 50" and the policy is pushed
  toward the correct integer.

  Exact answers still score exactly 1.0 so the summed reward (with the 0.1 format
  term) crosses the 1.0 "solved" threshold in :func:`metric_fn`; near-misses get
  at most 0.5 (so summed < 1.0 and ``solve_ratio`` stays a clean exact-match
  rate). Partial credit decays linearly to 0 at :data:`_PROXIMITY_SCALE` away.

  Args:
    prompts: the batch of prompt strings (unused).
    completions: the batch of decoded completion strings.
    answer: the batch of gold answer strings (forwarded dataset column).
    **kwargs: other forwarded dataset columns (unused).

  Returns:
    One float per completion: 1.0 exact, else ``0.5 * max(0, 1 - |pred-gold| /
    scale)``, else 0.0 if no integer parses.
  """
  del prompts, kwargs
  rewards: list[float] = []
  for completion, gold in zip(completions, answer):
    gold = str(np.asarray(gold).item()) if not isinstance(gold, str) else gold
    parsed = _parse_answer(str(completion))
    if parsed is None:
      rewards.append(0.0)
    elif parsed == int(gold):
      rewards.append(1.0)
    else:
      dist = abs(parsed - int(gold))
      rewards.append(0.5 * max(0.0, 1.0 - dist / _PROXIMITY_SCALE))
  return rewards


def metric_fn(prompts, completions, rewards, advantages, **kwargs) -> dict:
  """Reports solve-rate stats over the batch for logging / curriculum gating.

  Treats a completion as "solved" iff its (summed) reward is at least 1.0, i.e.
  ``answer_reward`` fired (the small format reward alone is < 1.0). Mirrors the
  frozenlake/toy_cats metric shape.

  Args:
    prompts: the batch of prompts (unused).
    completions: the batch of completions (unused).
    rewards: per-completion summed rewards.
    advantages: per-completion advantages (unused).
    **kwargs: forwarded dataset columns (unused).

  Returns:
    A dict of metric name -> ``(value, aggregation_fn)``.
  """
  del prompts, completions, advantages, kwargs
  rewards = np.asarray(rewards, dtype=np.float32)
  solved = rewards >= 1.0
  solve_ratio = float(solved.mean()) if solved.size else 0.0
  return {
      "arithmetic/solve_ratio": (solve_ratio, np.mean),
      "arithmetic/solve_all": (float(bool(solved.all())), np.mean),
      "arithmetic/solve_none": (float(bool((~solved).all())), np.mean),
  }
