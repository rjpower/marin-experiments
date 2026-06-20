# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test-case-graded, curriculum-scheduled multi-turn coding env (issue #8).

The redesign (see ``CURRICULUM_DESIGN.md``) that gives Dr.GRPO a real gradient:
the agent writes ``def solve(...)`` graded on N test cases, rewarded by the
**fraction of tests passed** (continuous -> intra-group reward variance), over a
**curriculum** of difficulty levels (:mod:`coding_problems`, :mod:`curriculum`).

This module reuses the agentic multi-turn machinery from
:mod:`coding_agent_env` (the program-as-tool parser/agent, the per-turn ``END``
stop, the rollout-identical prompt builder, the pass@k estimator) and swaps in:
  * a CoT "reason then write ``def solve``" few-shot prompt (breaks the
    empty-program collapse -- the policy always emits a full attempt);
  * a :class:`TestCaseEnvironment` that grades each round with
    :func:`coding_problems.grade_problem` and feeds the PUBLIC-test results back;
  * a curriculum-scheduled dataset (:func:`build_curriculum_dataset`) that draws
    each prompt's level from :class:`curriculum.Curriculum`;
  * per-level pass@1/pass@k eval on HELD-OUT instances
    (:func:`evaluate_problems_passk`) -- the headline measurement.

Import-safe on CPU; unit-tested (dependency-light) in
``tests/test_curriculum_env.py``.
"""

from __future__ import annotations

import dataclasses
import json
import random
import uuid
from typing import Any, Dict, List, Tuple

import numpy as np

import grain.python as grain

from environments.coding_agent_env import (
    PassKResult,
    PassKTaskRow,
    RunCodeAgent,
    RunCodeParser,
    _build_round_prompt,
    program_terminal_eos_tokens,
)
from environments.coding_env import _GRADE_MAX_STEPS, extract_program
from problems.coding_problems import (
    NUM_LEVELS,
    Problem,
    format_problem_prompt,
    format_test_feedback,
    grade_problem,
    load_eval_problems,
    problem_reward,
    reference_for,
    sample_problem,
)
from environments.curriculum import Curriculum, CurriculumConfig

from tunix.rl.agentic.environments.tool_environment import ToolEnvironment

# A best-across-rounds reward at/above this means some round passed ALL tests
# (full pass = 0.10 has_code + 0.20 ran_ok + 0.70 frac = 1.0; any partial < 1.0).
SOLVE_REWARD_THRESHOLD = 0.999


# ---------------------------------------------------------------------------
# CoT few-shot prompt: reason briefly, then write def solve, then END; read the
# public-test result and fix. Ends WITHOUT a trailing newline (invariant D).
# ---------------------------------------------------------------------------
CODE_SOLVE_SYSTEM_PROMPT = (
    "You are a Python coding assistant. For each task, think briefly about the "
    "approach as a short comment, then write a function def solve(...) and a line "
    "containing only END. You will then be shown how solve does on the public "
    "tests as a line starting with \"Tool result:\". If any test fails, write a "
    "corrected solve (again ending with END). Keep fixing until all tests pass.\n"
    "Write a function solve(a, b) that returns a + b.\n"
    "# add the two arguments and return the sum\n"
    "def solve(a, b):\n"
    "  return a + b\n"
    "END\n"
    "Tool result: public tests passed 2/2\n"
    "Write a function solve(nums) that returns the largest value in nums.\n"
    "# scan the list tracking the running maximum\n"
    "def solve(nums):\n"
    "  best = nums[0]\n"
    "  for x in nums:\n"
    "    if x > best:\n"
    "      best = x\n"
    "  return best\n"
    "END\n"
    "Tool result: public tests passed 2/2"
)


def build_solve_prompt(task_prompt: str) -> str:
  """The initial (round-1) rollout/eval prompt: CoT few-shot + the task text."""
  return f"{CODE_SOLVE_SYSTEM_PROMPT}\n{task_prompt}\n"


# ---------------------------------------------------------------------------
# Problem (de)serialization for the grain dataset / agentic task dict.
# ---------------------------------------------------------------------------


def problem_to_json(problem: Problem) -> str:
  """Serialize a Problem (incl. its tests) for the dataset ``answer`` column."""
  return json.dumps({
      "id": problem.id,
      "level": problem.level,
      "family": problem.family,
      "prompt": problem.prompt,
      "public_tests": [[list(a), e] for a, e in problem.public_tests],
      "hidden_tests": [[list(a), e] for a, e in problem.hidden_tests],
  })


def problem_from_json(s: str) -> Problem:
  """Reconstruct a Problem from :func:`problem_to_json` (args become tuples)."""
  d = json.loads(s)
  return Problem(
      id=d["id"],
      level=int(d["level"]),
      family=d["family"],
      prompt=d["prompt"],
      public_tests=[(tuple(a), e) for a, e in d["public_tests"]],
      hidden_tests=[(tuple(a), e) for a, e in d["hidden_tests"]],
  )


# ---------------------------------------------------------------------------
# Environment: grade each round on the test cases; score the best round.
# ---------------------------------------------------------------------------


class TestCaseEnvironment(ToolEnvironment):
  """``ToolEnvironment`` grading ``def solve`` against test cases per round.

  Each round the agent's program is graded by :func:`coding_problems.grade_problem`;
  the env stashes the best dense reward (:func:`coding_problems.problem_reward`)
  seen so far and feeds the PUBLIC-test results back as the tool output. The
  episode ends early when a round passes ALL tests, else at ``max_steps``. The
  trajectory reward is the **best-across-rounds** reward (zero-arg
  :meth:`_compute_final_reward`, added to the last step by the collector).
  """

  def __init__(
      self,
      task: Dict[str, Any] | None = None,
      *,
      tool_map=None,
      max_steps: int = 5,
      grade_max_steps: int = _GRADE_MAX_STEPS,
      **kwargs,
  ):
    self._problem = problem_from_json(task["answer"]) if task and task.get("answer") else None
    self._grade_max_steps = grade_max_steps
    self._best_reward = 0.0
    self._best_frac = 0.0
    self._best_exact = 0.0
    self._rounds_run = 0
    super().__init__(task=task, tool_map=tool_map or {}, reward_fn=None, max_steps=max_steps, **kwargs)
    self.final_reward_fn = self._compute_final_reward

  def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> Dict[str, str]:
    outputs: Dict[str, str] = {}
    for tc in tool_calls:
      call_id = tc.get("id") or str(uuid.uuid4())
      try:
        args = tc["function"]["arguments"]
        args = json.loads(args) if isinstance(args, str) else args
        source = args.get("source", "")
      except Exception:  # pylint: disable=broad-except
        source = ""
      if self._problem is None:
        outputs[call_id] = "(no problem)"
        continue
      components = grade_problem(source, self._problem, max_steps=self._grade_max_steps)
      reward = problem_reward(components)
      if reward > self._best_reward:
        self._best_reward = reward
        self._best_frac = components["frac_passed"]
        self._best_exact = components["exact"]
      self._rounds_run += 1
      outputs[call_id] = format_test_feedback(source, self._problem, max_steps=self._grade_max_steps)
    return outputs

  def _step_impl(self, action: Any):
    result = super()._step_impl(action)
    if self._best_exact >= 1.0:
      result = dataclasses.replace(result, done=True)
    return result

  def _compute_final_reward(self) -> float:
    return float(self._best_reward)


# ---------------------------------------------------------------------------
# Metric: first-attempt vs best-across-rounds, fraction-passed, mean level.
# ---------------------------------------------------------------------------


def solve_metric_fn(prompts, completions, rewards, advantages, answer, **kwargs) -> dict:
  """Logs solve/frac/level metrics for a Dr.GRPO step.

  ``rewards`` is the per-trajectory best-across-rounds reward. ``completions`` is
  the FIRST assistant turn, graded against the problem (carried in ``answer`` as
  JSON) to get the first-attempt fraction/solve. The gap between best-solve and
  first-solve is the multi-turn lift; first_frac is the dense first-attempt signal.
  """
  del prompts, advantages, kwargs
  rewards = np.asarray(rewards, dtype=np.float32)
  best_solved = rewards >= SOLVE_REWARD_THRESHOLD
  first_frac, first_solve, levels = [], [], []
  for completion, ans in zip(completions, answer):
    try:
      problem = problem_from_json(str(ans))
    except Exception:  # pylint: disable=broad-except
      continue
    comps = grade_problem(extract_program(str(completion)), problem)
    first_frac.append(comps["frac_passed"])
    first_solve.append(comps["exact"])
    levels.append(problem.level)
  mean = lambda xs: float(np.mean(xs)) if len(xs) else 0.0
  return {
      "coding/best_solve": (float(best_solved.mean()) if best_solved.size else 0.0, np.mean),
      "coding/first_solve": (mean(first_solve), np.mean),
      "coding/first_frac": (mean(first_frac), np.mean),
      "coding/reward_mean": (float(rewards.mean()) if rewards.size else 0.0, np.mean),
      "coding/mean_level": (mean(levels), np.mean),
  }


# ---------------------------------------------------------------------------
# SFT segments: Task -> def solve (the family reference) -> END.
# ---------------------------------------------------------------------------


def solve_segments(rng: random.Random, levels: Tuple[int, ...]):
  """One SFT transcript teaching the ``def solve`` format on an easy level.

  ``Task: ...`` line is context (mask 0); the reference ``def solve ... \\nEND\\n``
  is the model's turn (mask 1). Consumed by ``agentic_sft.run_sft_warmup`` with
  ``prompt_prefix=CODE_SOLVE_SYSTEM_PROMPT`` so SFT context == the RL prompt.
  """
  level = rng.choice(levels)
  problem = sample_problem(rng, level)
  reference = reference_for(problem)
  task = format_problem_prompt(problem)
  return [(f"{task}\n", 0), (f"{reference}\nEND\n", 1)]


# ---------------------------------------------------------------------------
# Curriculum-scheduled grain dataset.
# ---------------------------------------------------------------------------


class _CurriculumSource(grain.RandomAccessDataSource):
  """A grain source whose per-prompt level follows the curriculum schedule.

  We materialize the schedule deterministically: stepping a
  :class:`curriculum.Curriculum` once per training step and sampling
  ``batch_size`` prompt levels per step from its frontier-biased weights. With
  the default ``promote_threshold=0.0`` this is a pure fixed-cadence ramp (unlock
  the next level every ``steps_per_level`` steps); a higher threshold would gate
  on mastery once runtime success is wired in.
  """

  def __init__(self, steps: int, batch_size: int, seed: int, cur_config: CurriculumConfig):
    cur = Curriculum(cur_config)
    rng = random.Random(seed)
    self._rows: List[Tuple[str, str]] = []
    counter = seed * 2_654_435_761
    for step in range(steps + 1):  # +1 batch of slack (matches build_code_agent_dataset)
      for _ in range(batch_size):
        counter += 1
        level = cur.sample_level(counter & 0x7FFFFFFF)
        problem = sample_problem(rng, level)
        self._rows.append((format_problem_prompt(problem), problem_to_json(problem)))
      cur.on_step()

  def __len__(self) -> int:
    return len(self._rows)

  def __getitem__(self, idx: int) -> Tuple[str, str]:
    return self._rows[idx]


def build_curriculum_dataset(
    steps: int, batch_size: int, seed: int, cur_config: CurriculumConfig
) -> grain.MapDataset:
  """Batched grain dataset (``prompts`` task text + ``answer`` problem JSON)."""
  source = _CurriculumSource(steps, batch_size, seed, cur_config)

  def _to_columns(batch):
    prompts, answers = batch
    return {"prompts": prompts, "answer": answers}

  return grain.MapDataset.source(source).batch(batch_size).map(_to_columns)


# ---------------------------------------------------------------------------
# Per-level pass@k eval on held-out instances (the headline measurement).
# ---------------------------------------------------------------------------


def evaluate_problems_passk(
    model,
    tokenizer,
    problems: List[Problem],
    *,
    k: int = 16,
    max_new_tokens: int = 256,
    max_prompt_length: int = 1024,
    temperature: float = 1.0,
    cache_size: int | None = None,
    mesh=None,
    seed: int = 0,
) -> PassKResult:
  """First-attempt pass@k per problem (single ``def solve`` turn), tier = level.

  Samples ``k`` round-1 programs per problem at ``temperature`` (distinct seed per
  draw) and counts how many pass ALL tests (``exact``). The :class:`PassKResult`
  then gives the per-level pass@1..pass@k curve on held-out instances -- the
  measurement that decides whether RL has sharpened reliability.
  """
  import contextlib

  from tunix.generate import sampler as sampler_lib

  if cache_size is None:
    cache_size = max_prompt_length + max_new_tokens + 16
  cache_config = sampler_lib.CacheConfig(
      cache_size=cache_size,
      num_layers=model.config.num_layers,
      num_kv_heads=model.config.num_kv_heads,
      head_dim=model.config.head_dim,
  )
  sampler = sampler_lib.Sampler(transformer=model, tokenizer=tokenizer, cache_config=cache_config)
  eos_tokens = sorted(set([tokenizer.eos_token_id]) | set(program_terminal_eos_tokens(tokenizer)))

  prompts = [build_solve_prompt(format_problem_prompt(p)) for p in problems]
  n_correct = [0] * len(problems)
  frac_sum = [0.0] * len(problems)  # mean fraction-of-tests-passed (the dense signal)
  ran_sum = [0.0] * len(problems)   # mean fraction-of-tests-that-ran (vs empty/broken)

  ctx = mesh if mesh is not None else contextlib.nullcontext()
  with ctx:
    for draw in range(k):
      texts: List[str] = []
      chunk = 16
      for s in range(0, len(prompts), chunk):
        out = sampler(
            input_strings=prompts[s : s + chunk],
            max_generation_steps=max_new_tokens,
            max_prompt_length=cache_size - max_new_tokens - 4,
            echo=False,
            eos_tokens=eos_tokens,
            temperature=temperature,
            seed=seed + draw,
        )
        texts.extend(out.text)
      for i, completion in enumerate(texts):
        comps = grade_problem(extract_program(completion), problems[i])
        frac_sum[i] += comps["frac_passed"]
        ran_sum[i] += comps["ran_ok"]
        if comps["exact"] >= 1.0:
          n_correct[i] += 1

  # Per-level mean frac/ran -- the dense reward signal that decides if RL has a
  # gradient at the frontier (exact pass@k can be 0 while frac is partial).
  by_level: Dict[int, List[float]] = {}
  ran_by_level: Dict[int, List[float]] = {}
  for i, p in enumerate(problems):
    by_level.setdefault(p.level, []).append(frac_sum[i] / k)
    ran_by_level.setdefault(p.level, []).append(ran_sum[i] / k)
  cells = []
  for lvl in sorted(by_level):
    cells.append(f"L{lvl} frac={np.mean(by_level[lvl]):.3f} ran={np.mean(ran_by_level[lvl]):.3f}")
  print("[curric] FRAC-by-level (mean over draws): " + "  ".join(cells), flush=True)

  rows = [
      PassKTaskRow(task_id=p.id, tier=p.level, n_correct=n_correct[i], k=k)
      for i, p in enumerate(problems)
  ]
  return PassKResult(rows=rows, k=k, temperature=temperature)


def load_eval_suite(levels: Tuple[int, ...], n_per_level: int, seed: int) -> List[Problem]:
  """Held-out eval problems across ``levels`` (``n_per_level`` each)."""
  problems: List[Problem] = []
  for level in levels:
    problems.extend(load_eval_problems(level, n_per_level, seed))
  return problems

