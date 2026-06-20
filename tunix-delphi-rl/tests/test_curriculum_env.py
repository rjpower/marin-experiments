# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for the curriculum coding env (:mod:`environments.curriculum_env`).

Extracted verbatim from the module's former ``__main__`` self-check
(dependency-light, no TPU): problem (de)serialization, best-across-rounds env
grading, first-attempt metric, SFT segments, dataset columns + the curriculum
level ramp, and the round-prompt builder.
"""

from __future__ import annotations

import json
import random

import environments.curriculum_env as cev
from environments.curriculum import CurriculumConfig
from environments.curriculum_env import (
    CODE_SOLVE_SYSTEM_PROMPT,
    SOLVE_REWARD_THRESHOLD,
    _build_round_prompt,
    build_curriculum_dataset,
    problem_from_json,
    problem_to_json,
    solve_metric_fn,
    solve_segments,
)
# Aliased so pytest does not try to collect the env class as a test class.
from environments.curriculum_env import TestCaseEnvironment as _TestCaseEnvironment
from problems.coding_problems import (
    NUM_LEVELS,
    format_problem_prompt,
    reference_for,
    sample_problem,
)


def _call(src):
  return [{
      "id": "c1",
      "type": "function",
      "function": {"name": "run_code", "arguments": json.dumps({"source": src})},
  }]


def test_problem_json_roundtrip():
  rng = random.Random(0)
  prob = sample_problem(rng, 2)
  rt = problem_from_json(problem_to_json(prob))
  assert rt.family == prob.family and rt.hidden_tests == prob.hidden_tests


def test_env_grades_reference_vs_wrong():
  rng = random.Random(0)
  prob = sample_problem(rng, 2)
  ref = reference_for(prob)
  env = _TestCaseEnvironment(task={"answer": problem_to_json(prob)}, tool_map={}, max_steps=5)

  env._execute_tool_calls(_call("def solve(*a):\n  return 0"))
  assert 0.0 <= env._best_reward < SOLVE_REWARD_THRESHOLD, env._best_reward
  fb_ok = env._execute_tool_calls(_call(ref))
  assert env._best_exact == 1.0 and env._compute_final_reward() >= SOLVE_REWARD_THRESHOLD
  assert "public tests passed" in fb_ok["c1"], fb_ok


def test_solve_metric_first_attempt():
  rng = random.Random(0)
  prob = sample_problem(rng, 2)
  ref = reference_for(prob)
  m = solve_metric_fn(["t"], [ref + "\nEND"], [1.0], None, [problem_to_json(prob)])
  assert m["coding/first_solve"][0] == 1.0 and m["coding/best_solve"][0] == 1.0
  assert m["coding/mean_level"][0] == float(prob.level)


def test_solve_segments():
  # Task context (mask 0) then reference program (mask 1, END).
  segs = solve_segments(random.Random(1), (1, 2))
  assert segs[0][1] == 0 and segs[-1][1] == 1 and segs[-1][0].rstrip().endswith("END")


def test_dataset_columns_and_level_ramp():
  cfg = CurriculumConfig(num_levels=NUM_LEVELS, steps_per_level=2, promote_threshold=0.0)
  ds = build_curriculum_dataset(steps=10, batch_size=4, seed=0, cur_config=cfg)
  row = next(iter(ds))
  assert set(row.keys()) == {"prompts", "answer"}
  # Early prompts are level 1; later prompts include higher levels (the ramp).
  src = cev._CurriculumSource(steps=20, batch_size=4, seed=0, cur_config=cfg)
  early = max(problem_from_json(src[i][1]).level for i in range(4))
  late = max(problem_from_json(src[i][1]).level for i in range(len(src) - 4, len(src)))
  assert early == 1 and late > early, (early, late)


def test_round_prompt_builder_with_cot_prompt():
  rng = random.Random(0)
  prob = sample_problem(rng, 2)
  built = _build_round_prompt(CODE_SOLVE_SYSTEM_PROMPT, format_problem_prompt(prob), [])
  assert built.startswith(CODE_SOLVE_SYSTEM_PROMPT)
