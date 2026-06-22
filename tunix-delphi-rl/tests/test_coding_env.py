# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for the single-turn coding env (:mod:`environments.coding_env`).

Extracted verbatim from the module's former ``__main__`` self-check: every
family produces a valid bounded program; per-tier sampling yields a runnable
solution with non-empty gold; the parser/reward/metric/hint-stripper behave on
synthetic completions; and the oracle (reference solutions) solves the eval
ladder.
"""

from __future__ import annotations

import random

import environments.coding_env as ce
import environments.micropython as micropython
import problems.coding_tasks as coding_tasks
from environments.coding_env import (
    code_metric_fn,
    code_reward,
    evaluate_completions,
    extract_program,
    sample_task,
    strip_answer_hint,
)


def test_every_family_produces_valid_program():
  rng = random.Random(0)
  bad = []
  for fam in ce.FAMILIES:
    for _ in range(20):
      prompt, solution = fam.sample(rng)
      res = micropython.run(solution, max_steps=ce._EVAL_MAX_STEPS)
      if not res.ok or not res.stdout:
        bad.append(f"{fam.id}: {res.error!r} prompt={prompt!r}")
  assert not bad, f"{len(bad)} family sample(s) produced an invalid program: {bad}"


def test_sample_task_per_tier_runs_to_gold():
  rng = random.Random(0)
  for tier in (0, 1, 2, 3, 4, 5):
    for _ in range(20):
      prompt, solution, gold = sample_task(rng, (tier,))
      assert gold, f"tier {tier}: empty gold for prompt={prompt!r}"
      assert micropython.run(solution, max_steps=ce._EVAL_MAX_STEPS).ok


def test_parser_reward_and_metric():
  # Parser + reward on a synthetic 'good' completion.
  good = "print(6 * 7)\nEND\nTask: something else\n"
  assert extract_program(good) == "print(6 * 7)"
  assert code_reward(["p"], [good], ["42\n"])[0] == 2.0
  assert code_metric_fn(["p"], [good], None, None, ["42\n"])["coding/solve_ratio"][0] == 1.0
  # A close-but-wrong program earns partial but not solve.
  near = "print(41)\nEND\n"
  r_near = code_reward(["p"], [near], ["42\n"])[0]
  assert 0.4 <= r_near < 1.3, r_near
  assert code_metric_fn(["p"], [near], None, None, ["42\n"])["coding/solve_ratio"][0] == 0.0


def test_hint_stripping_keeps_input_drops_output_leak():
  assert strip_answer_hint("Print fib(10). (The answer is 55.)", "55\n") == "Print fib(10)."
  assert strip_answer_hint(
      "Print the sum from 1 to 100 (inclusive).", "5050\n"
  ) == "Print the sum from 1 to 100 (inclusive)."


def test_oracle_reference_solutions_all_solve():
  tasks = coding_tasks.load_tasks()
  oracle = evaluate_completions(tasks, [t.solution + "\nEND" for t in tasks])
  assert oracle.solved == oracle.total, "reference solutions must all solve"
