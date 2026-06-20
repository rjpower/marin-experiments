# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for the multi-turn coding env (:mod:`environments.coding_agent_env`).

Extracted verbatim from the module's former ``__main__`` self-check
(dependency-light, no TPU): the program parser, best-across-rounds env grading,
first-vs-best metric, SFT segments, the round-prompt builder, dataset columns,
eval-prompt composition, and the pass@k unbiased estimator.
"""

from __future__ import annotations

import json
import random

from environments.coding_agent_env import (
    CODE_AGENT_SYSTEM_PROMPT,
    RUN_CODE_TOOL_NAME,
    SOLVE_REWARD_THRESHOLD,
    CodeRunEnvironment,
    PassKResult,
    PassKTaskRow,
    RunCodeParser,
    _build_round_prompt,
    build_agent_prompt,
    build_code_agent_dataset,
    code_agent_metric_fn,
    code_agent_segments,
)
from environments.coding_env import strip_answer_hint
from problems.coding_tasks import load_tasks


def test_run_code_parser():
  # A program turn -> a run_code call; an empty turn -> finish ([]).
  p = RunCodeParser()
  calls = p.parse("print(6 * 7)\nEND\nTool result: 42\n")
  assert calls and calls[0].name == RUN_CODE_TOOL_NAME
  assert calls[0].arguments["source"] == "print(6 * 7)"
  assert p.parse("\n\n") == []


def _call(src):
  return [{
      "id": "c1",
      "type": "function",
      "function": {"name": RUN_CODE_TOOL_NAME, "arguments": json.dumps({"source": src})},
  }]


def test_env_best_across_rounds_reward():
  env = CodeRunEnvironment(task={"answer": "42\n"}, tool_map={}, max_steps=5)
  fb1 = env._execute_tool_calls(_call("print(41)"))  # wrong but runs
  assert "41" in fb1["c1"], fb1
  assert 0.0 < env._best_reward < SOLVE_REWARD_THRESHOLD
  fb2 = env._execute_tool_calls(_call("print(6 * 7)"))  # exact
  assert fb2["c1"] == "42"
  assert env._best_components["exact"] == 1.0
  assert env._compute_final_reward() >= SOLVE_REWARD_THRESHOLD
  # An error feedback is surfaced for a crashing program.
  env2 = CodeRunEnvironment(task={"answer": "1\n"}, tool_map={}, max_steps=5)
  fb_err = env2._execute_tool_calls(_call("print(undefined_name)"))
  assert fb_err["c1"].startswith("Error:"), fb_err


def test_first_vs_best_metric():
  m = code_agent_metric_fn(
      ["Task: x"], ["print(6 * 7)\nEND"], [2.0], None, ["42\n"]
  )
  assert m["coding/solve_ratio"][0] == 1.0
  assert m["coding/first_solve"][0] == 1.0
  m2 = code_agent_metric_fn(
      ["Task: x"], ["print(41)\nEND"], [2.0], None, ["42\n"]
  )
  assert m2["coding/solve_ratio"][0] == 1.0  # best (the trajectory) solved
  assert m2["coding/first_solve"][0] == 0.0  # but round-1 did not -> the lift


def test_sft_segments_program_and_fix_demos():
  rng = random.Random(0)
  any_fix = False
  for _ in range(200):
    segs = code_agent_segments(rng, (0, 1, 2, 3, 4), fix_prob=0.5)
    assert segs[0][1] == 0 and segs[0][0].startswith("Task: ")
    assert segs[-1][1] == 1 and segs[-1][0].rstrip().endswith("END")
    if len(segs) > 2:
      any_fix = True
      assert segs[2][0].startswith("Tool result: ")
  assert any_fix, "expected some fix transcripts at fix_prob=0.5"


def test_round_prompt_builder():
  built = _build_round_prompt(
      CODE_AGENT_SYSTEM_PROMPT, "Print 5.", [("print(4)", "4"), ("print(5)", "5")]
  )
  assert built.startswith(CODE_AGENT_SYSTEM_PROMPT + "\nTask: Print 5.\n")
  assert "print(4)\nEND\nTool result: 4\n" in built
  assert built.endswith("print(5)\nEND\nTool result: 5\n")


def test_dataset_columns_and_eval_prompt():
  ds = build_code_agent_dataset(n=8, seed=0, batch_size=4, tiers=(0, 1, 2, 3, 4))
  row = next(iter(ds))
  assert set(row.keys()) == {"prompts", "answer"}

  tasks = load_tasks()
  ep = build_agent_prompt(strip_answer_hint(tasks[0].prompt, tasks[0].answer))
  assert ep.startswith(CODE_AGENT_SYSTEM_PROMPT)


def test_passk_unbiased_estimator():
  # Monotone non-decreasing in m, exact at the corners.
  pk = PassKResult(
      rows=[
          PassKTaskRow("a", 5, n_correct=0, k=8),   # never solved
          PassKTaskRow("b", 5, n_correct=8, k=8),   # always solved
          PassKTaskRow("c", 5, n_correct=1, k=8),   # rare-but-present (the RL regime)
      ],
      k=8,
      temperature=1.0,
  )
  assert abs(pk.pass_at(1) - (0.0 + 1.0 + 1 / 8) / 3) < 1e-9, pk.pass_at(1)
  assert abs(pk.pass_at(8) - (0.0 + 1.0 + 1.0) / 3) < 1e-9, pk.pass_at(8)
  assert pk.pass_at(1) <= pk.pass_at(2) <= pk.pass_at(4) <= pk.pass_at(8)
  assert PassKResult._pass_at_m_one(8, 1, 1) == 0.125
  assert PassKResult._pass_at_m_one(8, 1, 8) == 1.0
  assert "pass@1=" in pk.summary() and "tier 5" in pk.summary()
