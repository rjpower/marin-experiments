# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Multi-turn agentic-CODING environment for Delphi (issue #8).

Escalates the single-turn coding setup (:mod:`coding_env`) to a **multi-turn
write -> run -> read-output -> revise** loop on tunix's *agentic* GRPO stack
(``tunix.rl.agentic``), the same multi-turn machinery the CALC tool stages used
(:mod:`agentic_tools`). The agent gets up to ``max_rounds`` rounds per task:

  1. it writes a Python program terminated by a line containing only ``END``;
  2. the :mod:`micropython` interpreter (the "tool") executes it and the env
     feeds the program's **stdout** (or its error) back as a ``Tool result:``
     message;
  3. the agent reads that output and, if it is wrong or errored, writes a
     corrected program -- repeating until the output is right or the rounds run
     out. The episode ends early the moment a program's stdout exactly matches
     the gold (success), or at ``max_rounds``.

Why this is the regime where RL should finally beat SFT (cf. issue #7, where SFT
saturated the single-turn ladder and Dr.GRPO was marginal): on a *harder* task
distribution the base LM's FIRST attempt frequently errors or is off-by-one, and
the winning behavior is to read the execution feedback and FIX it. That repair
behavior is only weakly demonstrable by SFT (we make it in-distribution but rare
-- see :func:`code_agent_segments`); amplifying a rare-but-present behavior from
the base policy is exactly what RL is for. So the headline metric is the gap
between the **first-attempt** solve rate and the **best-across-rounds** solve
rate, and how RL grows the latter.

How the multi-turn reward flows (verified against the installed tunix agentic
source): the learner's per-completion ``reward_fns`` only see the *first*
assistant turn, so the multi-round grade lives in the ENVIRONMENT instead. The
trajectory reward is the undiscounted sum of per-step rewards plus the env's
zero-arg ``final_reward_fn`` (added to the last step by the trajectory collector);
we make every per-step reward 0 and return the dense **best-across-rounds** grade
from :meth:`CodeRunEnvironment._compute_final_reward`. The learner is created
with ``reward_fns=None`` so the reward IS that trajectory reward.

Per-turn stop: programs are multi-line, so (unlike the single-line CALC turns) we
cannot stop on a newline. Instead each turn stops on the ``END`` sentinel token
(:func:`program_terminal_eos_tokens`); without a per-turn stop the first turn
would consume the whole per-episode response budget.

This module is import-safe on CPU; its dependency-light unit tests (parser / env
grade / SFT segments / the greedy multi-turn eval string builder, no TPU) live in
``tests/test_coding_agent_env.py``.
"""

from __future__ import annotations

import dataclasses
import json
import random
import re
import uuid
from typing import Any, Dict, List, Tuple

import numpy as np

import grain.python as grain

import environments.micropython as micropython
from environments.coding_env import (
    _EVAL_MAX_STEPS,
    _GRADE_MAX_STEPS,
    _coerce_gold,
    _output_similarity,
    extract_program,
    grade_program,
    families_for_tiers,
    sample_task,
    strip_answer_hint,
)
from problems.coding_tasks import load_tasks
from models.delphi_qwen3 import DELPHI_EOS_ID

from tunix.rl.agentic.agents import base_agent
from tunix.rl.agentic.agents.tool_agent import ToolAgent
from tunix.rl.agentic.environments.tool_environment import ToolEnvironment
from tunix.rl.agentic.tools import base_tool

ToolCall = base_tool.ToolCall

# Name of the single "run this program" tool. The parser maps every program the
# model emits to one such call; the env executes it via micropython.
RUN_CODE_TOOL_NAME = "run_code"

# Dense reward weights (identical to the single-turn coding reward in
# coding_env): exact match dominates, but has_code / ran_ok / output-overlap give
# a smooth climb so a Dr.GRPO group has non-zero advantage before any round is
# exact. An exact program scores 2.0; a runs-but-wrong program scores 0.4..1.0.
_W_HAS_CODE = 0.1
_W_RAN_OK = 0.3
_W_PARTIAL = 0.6
_W_EXACT = 1.0

# A best-across-rounds reward at or above this threshold means some round was
# exact (the maximum NON-exact reward is 0.1 + 0.3 + 0.6 = 1.0, reached only in
# the limit partial->1 which implies exact, so any real non-exact reward < 1.0).
SOLVE_REWARD_THRESHOLD = 1.5

# How much program-output feedback to show the model per round (chars).
_MAX_FEEDBACK_CHARS = 400


# ---------------------------------------------------------------------------
# Few-shot prompt: teaches the multi-turn write -> END -> read -> revise format.
# ---------------------------------------------------------------------------
#
# Two demos. The first is a clean single-round success (the format). The second
# shows a genuine off-by-one FIX: the first attempt prints "1,2,3,4" for a "1 to
# 5" task, the tool result reveals the short output, and the corrected program
# uses the right range bound. This is what makes "after a wrong Tool result,
# write a corrected program" in-distribution. Ends WITHOUT a trailing newline so
# the rollout/eval prompt builder and the SFT prefix append exactly one "\n"
# (train/RL prompt match, invariant D in AGENTS.md).
CODE_AGENT_SYSTEM_PROMPT = (
    "You are a Python coding assistant. For each task, write a short Python "
    "program and then a line containing only END. You will then be shown the "
    "program's output as a line starting with \"Tool result:\". If the output "
    "is wrong or there is an error, write a corrected program (again ending "
    "with END). Keep fixing until the output is right.\n"
    "Task: Print the sum of the integers from 1 to 4 (inclusive).\n"
    "total = 0\n"
    "for i in range(1, 5):\n"
    "  total += i\n"
    "print(total)\n"
    "END\n"
    "Tool result: 10\n"
    "Task: Print the numbers from 1 to 5 (inclusive) separated by commas, with "
    "no spaces.\n"
    "print(','.join([str(i) for i in range(1, 5)]))\n"
    "END\n"
    "Tool result: 1,2,3,4\n"
    "print(','.join([str(i) for i in range(1, 6)]))\n"
    "END\n"
    "Tool result: 1,2,3,4,5"
)


def build_agent_prompt(task_prompt: str) -> str:
  """The initial (round-1) rollout/eval prompt: few-shot + the target task line.

  ``CODE_AGENT_SYSTEM_PROMPT + "\\n"`` matches what the SFT warm-up prepends when
  given ``prompt_prefix=CODE_AGENT_SYSTEM_PROMPT`` and what the agentic chat
  parser renders for [system, user] with ``generation_suffix="\\n"``, so the SFT
  context == the RL rollout prompt == this eval prompt.
  """
  return f"{CODE_AGENT_SYSTEM_PROMPT}\nTask: {task_prompt}\n"


def program_terminal_eos_tokens(tokenizer) -> List[int]:
  """Token ids that mark the end of a program turn (the ``END`` sentinel).

  Programs are multi-line, so a per-turn turn-boundary stop cannot be a newline
  (that would cut after the first line). Instead the model ends each program with
  a line containing only ``END``; we stop generation on any token whose decoded
  text, stripped of surrounding whitespace, is exactly ``END`` (this covers
  ``"END"``, ``" END"``, ``"END\\n"``, ``" END\\n"`` and ``"\\nEND"`` on the
  Llama-3 BPE). Returns the sorted id list; raising if empty is the caller's job
  (an empty set would mean the first turn never stops).
  """
  ids: List[int] = []
  for tid in range(int(tokenizer.vocab_size)):
    text = tokenizer.decode([tid])
    if text.strip() == "END":
      ids.append(tid)
  return sorted(ids)


def _format_feedback(result: "micropython.ExecResult", source: str) -> str:
  """The ``Tool result:`` body shown to the model for one executed program."""
  if not source.strip():
    return "(no program)"
  if not result.ok:
    return f"Error: {result.error}"[:_MAX_FEEDBACK_CHARS]
  if not result.stdout:
    return "(no output)"
  text = result.stdout
  if len(text) > _MAX_FEEDBACK_CHARS:
    text = text[:_MAX_FEEDBACK_CHARS] + "...(truncated)"
  # Render the trailing newline of print() visibly-safe: keep as-is so the
  # multi-line outputs (one number per line) match the few-shot demos.
  return text.rstrip("\n")


def _grade_components(result: "micropython.ExecResult", source: str, gold: str) -> Dict[str, float]:
  """has_code / ran_ok / exact / partial for one executed program (cf. grade_program)."""
  if not source.strip():
    return {"has_code": 0.0, "ran_ok": 0.0, "exact": 0.0, "partial": 0.0}
  ran_ok = 1.0 if result.ok else 0.0
  exact = 1.0 if (result.ok and result.stdout == gold) else 0.0
  partial = _output_similarity(result.stdout, gold) if result.ok else 0.0
  return {"has_code": 1.0, "ran_ok": ran_ok, "exact": exact, "partial": partial}


def _reward_of(components: Dict[str, float]) -> float:
  """The dense reward for one program's grade components."""
  return (
      _W_HAS_CODE * components["has_code"]
      + _W_RAN_OK * components["ran_ok"]
      + _W_PARTIAL * components["partial"]
      + _W_EXACT * components["exact"]
  )


# ---------------------------------------------------------------------------
# Parser: every emitted program becomes a run_code call; empty => finish.
# ---------------------------------------------------------------------------


class RunCodeParser:
  """Parses a model turn into a single ``run_code`` :class:`ToolCall`.

  Duck-types the tool-parser interface :class:`ToolAgent` uses. ``parse`` returns
  one ``run_code`` call carrying the extracted program as ``{"source": ...}``
  whenever the turn contains a non-empty program (so the episode CONTINUES and the
  env runs it); an empty turn returns ``[]`` so ``ToolAgent.update_from_model``
  treats it as a ``finish`` (the episode ends, scored on the best round so far).
  ``get_tool_prompt`` is suppressed -- a base LM must not see tool prose; the
  few-shot :data:`CODE_AGENT_SYSTEM_PROMPT` carries the format.
  """

  def get_tool_prompt(self, tools=None, schema_style: str = "openai") -> str:
    del tools, schema_style
    return ""

  def parse(self, model_response: str) -> list:
    program = extract_program(model_response or "")
    if not program.strip():
      return []
    return [ToolCall(name=RUN_CODE_TOOL_NAME, arguments={"source": program})]


# ---------------------------------------------------------------------------
# Agent: suppressed tool docs, task-as-user-turn, raw tool-result rendering,
# and END re-appended to the recorded assistant turn so the multi-turn CONTEXT
# matches the SFT transcripts (the rollout strips the END stop token).
# ---------------------------------------------------------------------------


class RunCodeAgent(ToolAgent):
  """A :class:`ToolAgent` for the multi-turn coder (cf. agentic_tools.DelphiToolAgent)."""

  def __init__(self, system_prompt: str):
    from tunix.rl.agentic.tools import tool_manager as _tool_manager

    self.tool_manager = _tool_manager.ToolManager(tool_map={})
    self.tool_parser = RunCodeParser()
    self.tools_prompt = ""
    base_agent.ConversationAgentBase.__init__(self, system_prompt=system_prompt)

  def _observation_to_messages(
      self, observation: Any, reward: float, done: bool, info: Dict[str, Any]
  ) -> None:
    del reward, done, info
    if isinstance(observation, dict):
      # Initial task observation: the dataset row, keyed "prompts".
      if "prompts" in observation:
        content = observation["prompts"]
        self._messages.append(
            {"role": "user", "content": "" if content is None else str(content)}
        )
        return
      # Tool-output observation: inject the RAW program output (the chat parser
      # adds the "Tool result: " prefix, matching the few-shot demos).
      if "tool_outputs" in observation:
        for call_id, output in observation["tool_outputs"].items():
          self._messages.append({
              "role": "tool",
              "tool_call_id": call_id,
              "content": output or "",
          })
        return
      if not observation:  # terminal step: {}.
        return
    super()._observation_to_messages(observation, 0.0, False, {})

  def update_from_model(self, response: str, **kwargs):
    """As ToolAgent.update_from_model, but records the assistant turn WITH ``END``.

    The rollout strips the ``END`` stop token from ``response``; re-appending a
    normalized ``\\nEND`` to the recorded assistant content makes the rendered
    multi-turn context (what later turns condition on) match the SFT transcripts,
    which train ``...program\\nEND\\n`` (invariant D). Training tokens/masks are
    unaffected (those come from the rollout's own token ids, not this text).
    """
    import copy as _copy

    from tunix.rl.agentic.agents import agent_types

    try:
      tool_calls = self.tool_parser.parse(response)
    except Exception:  # pylint: disable=broad-except
      tool_calls = []

    if not tool_calls:
      tool_calls_dict = [{
          "id": str(uuid.uuid4()),
          "type": "function",
          "function": {"name": "finish", "arguments": {"response": response}},
      }]
    else:
      tool_calls_dict = []
      for tool_call in tool_calls:
        args = tool_call.arguments
        if isinstance(args, dict):
          args = json.dumps(args)
        tool_calls_dict.append({
            "id": str(uuid.uuid4()),
            "type": "function",
            "function": {"name": tool_call.name, "arguments": args},
        })

    recorded = response if response.rstrip().endswith("END") else (
        response.rstrip("\n") + "\nEND"
    )
    self._messages.append({"role": "assistant", "content": recorded})
    step = agent_types.Step(
        chat_completions=_copy.deepcopy(self._messages),
        action=agent_types.Action(action=tool_calls_dict),
        model_response=response,
    )
    self._trajectory.steps.append(step)
    return agent_types.Action(action=tool_calls_dict)


# ---------------------------------------------------------------------------
# Environment: run each program, stash the best grade, score at episode end.
# ---------------------------------------------------------------------------


class CodeRunEnvironment(ToolEnvironment):
  """``ToolEnvironment`` that runs the agent's programs and scores the best round.

  Each round the agent's program is executed by micropython; the env stashes the
  best dense grade seen so far and feeds the program's stdout/error back as the
  tool output. The episode ends early on an exact match, else at ``max_steps``
  rounds. The terminal (trajectory) reward is the **best-across-rounds** dense
  grade, returned by the zero-arg :meth:`_compute_final_reward` that the
  trajectory collector adds to the last step.
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
    self._gold = _coerce_gold(task.get("answer", "")) if task else ""
    self._grade_max_steps = grade_max_steps
    self._best_components = {"has_code": 0.0, "ran_ok": 0.0, "exact": 0.0, "partial": 0.0}
    self._best_reward = 0.0
    self._rounds_run = 0
    super().__init__(task=task, tool_map=tool_map or {}, reward_fn=None, max_steps=max_steps, **kwargs)
    # The trajectory collector calls this zero-arg fn at episode end and adds it
    # to the last step's reward (=> the trajectory reward).
    self.final_reward_fn = self._compute_final_reward

  def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> Dict[str, str]:
    """Runs each program, updates the best grade, returns the per-call feedback."""
    outputs: Dict[str, str] = {}
    for tc in tool_calls:
      call_id = tc.get("id") or str(uuid.uuid4())
      try:
        args = tc["function"]["arguments"]
        args = json.loads(args) if isinstance(args, str) else args
        source = args.get("source", "")
      except Exception:  # pylint: disable=broad-except
        source = ""
      result = micropython.run(source, max_steps=self._grade_max_steps)
      components = _grade_components(result, source, self._gold)
      reward = _reward_of(components)
      if reward > self._best_reward:
        self._best_reward = reward
        self._best_components = components
      self._rounds_run += 1
      outputs[call_id] = _format_feedback(result, source)
    return outputs

  def _step_impl(self, action: Any):
    """As ToolEnvironment, but ends the episode the moment a round is exact."""
    result = super()._step_impl(action)
    if self._best_components["exact"] >= 1.0:
      result = dataclasses.replace(result, done=True)
    return result

  def _compute_final_reward(self) -> float:
    return float(self._best_reward)


# ---------------------------------------------------------------------------
# Metric: log the first-attempt vs best-across-rounds solve gap (the RL lift).
# ---------------------------------------------------------------------------


def code_agent_metric_fn(prompts, completions, rewards, advantages, answer, **kwargs) -> dict:
  """Logs first-attempt and best-across-rounds solve rates + mean reward.

  ``rewards`` is the per-trajectory best-across-rounds reward (the env terminal
  reward; no learner reward_fns are summed). ``completions`` is the FIRST
  assistant turn (round-1 program), so grading it gives the first-attempt rate.
  The gap ``solve_ratio - first_solve`` is the multi-turn lift RL should grow.
  """
  del prompts, advantages, kwargs
  rewards = np.asarray(rewards, dtype=np.float32)
  solved = rewards >= SOLVE_REWARD_THRESHOLD
  first_exact, first_ran = [], []
  for completion, gold in zip(completions, answer):
    g = _coerce_gold(gold)
    s = grade_program(extract_program(str(completion)), g)
    first_exact.append(s["exact"])
    first_ran.append(s["ran_ok"])
  mean = lambda xs: float(np.mean(xs)) if len(xs) else 0.0
  return {
      "coding/solve_ratio": (float(solved.mean()) if solved.size else 0.0, np.mean),
      "coding/first_solve": (mean(first_exact), np.mean),
      "coding/first_ran_ok": (mean(first_ran), np.mean),
      "coding/reward_mean": (float(rewards.mean()) if rewards.size else 0.0, np.mean),
  }


# ---------------------------------------------------------------------------
# SFT (multi-turn execution-format) transcripts, with a minority of fix demos.
# ---------------------------------------------------------------------------

# Mutations that turn a correct solution into a plausible "runs-but-wrong" (or
# crashing) first attempt: off-by-one range bounds, swapped operators/comparisons,
# and +/-1 integer literals. The first applicable one whose output differs from
# the gold (or which errors) is used.
_RANGE_RE = re.compile(r"range\((\s*-?\d+\s*),(\s*-?\d+\s*)\)")


def _mutate_to_bug(
    rng: random.Random, solution: str, gold: str
) -> Tuple[str, str] | None:
  """Returns (buggy_source, feedback_text) for an SFT fix demo, or None.

  Tries a few small mutations and keeps the first whose micropython result
  DIFFERS from the gold -- either a wrong stdout or an error -- so the demo's
  first attempt is genuinely wrong and the second (the real solution) is the fix.
  """
  candidates: List[str] = []
  # Off-by-one on the upper bound of the first range(a, b).
  m = _RANGE_RE.search(solution)
  if m:
    a, b = m.group(1), m.group(2)
    candidates.append(solution[: m.start()] + f"range({a},{int(b) - 1})" + solution[m.end():])
  # Operator / comparison swaps (first occurrence).
  for src, dst in ((" + ", " - "), (" - ", " + "), (" * ", " + "),
                   (" >= ", " > "), (" <= ", " < "), (" == ", " != ")):
    if src in solution:
      candidates.append(solution.replace(src, dst, 1))
  rng.shuffle(candidates)
  for cand in candidates:
    if cand == solution:
      continue
    res = micropython.run(cand, max_steps=_GRADE_MAX_STEPS)
    differs = (not res.ok) or (res.stdout != gold)
    if differs:
      feedback = f"Error: {res.error}" if not res.ok else (res.stdout.rstrip("\n") or "(no output)")
      return cand, feedback[:_MAX_FEEDBACK_CHARS]
  return None


def code_agent_segments(
    rng: random.Random, tiers: Tuple[int, ...], fix_prob: float = 0.3
):
  """One multi-turn SFT transcript: ``Task -> [buggy -> Tool result ->] solution``.

  Consumed by :func:`agentic_sft.run_sft_warmup` with
  ``prompt_prefix=CODE_AGENT_SYSTEM_PROMPT``. The ``Task:`` line and every
  ``Tool result:`` line are CONTEXT (mask 0 -- the env emits them at RL time);
  the programs (each ``...\\nEND\\n``) are the model's turns (mask 1). With
  probability ``fix_prob`` the transcript shows a buggy first attempt, its (wrong)
  tool result, then the corrected program -- making the read-output-and-fix
  behavior in-distribution but RARE, the regime RL amplifies.
  """
  prompt, solution, gold = sample_task(rng, tiers)
  segments = [(f"Task: {prompt}\n", 0)]
  if rng.random() < fix_prob:
    bug = _mutate_to_bug(rng, solution, gold)
    if bug is not None:
      buggy_src, buggy_feedback = bug
      segments.append((f"{buggy_src}\nEND\n", 1))
      segments.append((f"Tool result: {buggy_feedback}\n", 0))
  segments.append((f"{solution}\nEND\n", 1))
  return segments


# ---------------------------------------------------------------------------
# The grain dataset (prompts = "Task: ..." user turn; answer = gold stdout).
# ---------------------------------------------------------------------------


class _CodeAgentSource(grain.RandomAccessDataSource):
  """A grain source of ``(user_turn, gold)`` rows for the multi-turn coder."""

  def __init__(self, n: int, seed: int, tiers: Tuple[int, ...]):
    rng = random.Random(seed)
    self._rows: List[Tuple[str, str]] = []
    for _ in range(n):
      prompt, _solution, gold = sample_task(rng, tiers)
      self._rows.append((f"Task: {prompt}", gold))

  def __len__(self) -> int:
    return len(self._rows)

  def __getitem__(self, idx: int) -> Tuple[str, str]:
    return self._rows[idx]


def build_code_agent_dataset(
    n: int, seed: int, batch_size: int, tiers: Tuple[int, ...]
) -> grain.MapDataset:
  """Builds the batched grain dataset (``prompts`` user turn + ``answer`` gold)."""
  source = _CodeAgentSource(n, seed, tiers)

  def _to_columns(batch):
    prompts, answers = batch
    return {"prompts": prompts, "answer": answers}

  return grain.MapDataset.source(source).batch(batch_size).map(_to_columns)


# ---------------------------------------------------------------------------
# Greedy multi-turn evaluation on the fixed tasks (the headline measurement).
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MultiTurnTaskRow:
  """The multi-turn eval outcome for one fixed task."""

  task_id: str
  tier: int
  first_solved: bool
  best_solved: bool
  rounds_used: int
  final_program: str
  final_output: str


@dataclasses.dataclass
class MultiTurnEvalResult:
  """Aggregate multi-turn eval: first-attempt vs best-across-rounds solve."""

  rows: List[MultiTurnTaskRow]

  @property
  def total(self) -> int:
    return len(self.rows)

  @property
  def first_solved(self) -> int:
    return sum(1 for r in self.rows if r.first_solved)

  @property
  def best_solved(self) -> int:
    return sum(1 for r in self.rows if r.best_solved)

  def per_tier(self) -> Dict[int, Tuple[int, int, int]]:
    """tier -> (first_solved, best_solved, total)."""
    out: Dict[int, List[int]] = {}
    for r in self.rows:
      acc = out.setdefault(r.tier, [0, 0, 0])
      acc[0] += int(r.first_solved)
      acc[1] += int(r.best_solved)
      acc[2] += 1
    return {t: (f, b, n) for t, (f, b, n) in out.items()}

  def summary(self) -> str:
    lines = [
        f"first {self.first_solved}/{self.total} -> best {self.best_solved}/{self.total}"
    ]
    for tier in sorted(self.per_tier()):
      f, b, n = self.per_tier()[tier]
      lines.append(f"  tier {tier}: first={f}/{n} best={b}/{n}")
    return "\n".join(lines)


def _build_round_prompt(system_prompt: str, task_prompt: str, history: List[Tuple[str, str]]) -> str:
  """Reconstructs the rollout-identical conversation string for the next turn.

  ``history`` is the list of ``(program, feedback)`` pairs from prior rounds. The
  rendered string is ``<few-shot>\\nTask: <prompt>\\n`` followed by, per prior
  round, ``<program>\\nEND\\nTool result: <feedback>\\n`` -- byte-identical to what
  the agentic chat parser renders for the live rollout (system + user +
  assistant(``program\\nEND``) + tool(``Tool result: feedback``), joined by "\\n",
  with ``generation_suffix="\\n"``).
  """
  parts = [f"{system_prompt}\nTask: {task_prompt}\n"]
  for program, feedback in history:
    prog = program if program.rstrip().endswith("END") else program.rstrip("\n") + "\nEND"
    parts.append(f"{prog}\nTool result: {feedback}\n")
  return "".join(parts)


def evaluate_tasks_multiturn(
    model,
    tokenizer,
    tasks,
    *,
    max_rounds: int = 5,
    max_new_tokens: int = 192,
    max_prompt_length: int = 512,
    cache_size: int | None = None,
    mesh=None,
) -> MultiTurnEvalResult:
  """Greedy multi-turn eval: run write->run->revise for each task, lockstep by round.

  For each round, all not-yet-solved tasks are batched through the greedy sampler
  (temperature 0), their programs executed by micropython, and the output fed back
  into the next round's prompt. Records, per task, whether the FIRST attempt
  solved, whether ANY round solved (best), and how many rounds were used. This is
  the held-out measurement of the multi-turn lift; it mirrors the RL rollout's
  prompt rendering and per-turn ``END`` stop exactly.
  """
  import contextlib

  from tunix.generate import sampler as sampler_lib

  if cache_size is None:
    cache_size = max_prompt_length + max_rounds * (max_new_tokens + 64) + 16
  cache_config = sampler_lib.CacheConfig(
      cache_size=cache_size,
      num_layers=model.config.num_layers,
      num_kv_heads=model.config.num_kv_heads,
      head_dim=model.config.head_dim,
  )
  sampler = sampler_lib.Sampler(
      transformer=model, tokenizer=tokenizer, cache_config=cache_config
  )
  eos_tokens = sorted(set([DELPHI_EOS_ID]) | set(program_terminal_eos_tokens(tokenizer)))

  # Per-task running state.
  prompts = [strip_answer_hint(t.prompt, t.answer) for t in tasks]
  golds = [t.answer for t in tasks]
  history: List[List[Tuple[str, str]]] = [[] for _ in tasks]
  first_solved = [False] * len(tasks)
  best_solved = [False] * len(tasks)
  rounds_used = [0] * len(tasks)
  final_program = [""] * len(tasks)
  final_output = [""] * len(tasks)
  done = [False] * len(tasks)

  ctx = mesh if mesh is not None else contextlib.nullcontext()
  with ctx:
    for round_idx in range(max_rounds):
      active = [i for i in range(len(tasks)) if not done[i]]
      if not active:
        break
      batch_prompts = [
          _build_round_prompt(CODE_AGENT_SYSTEM_PROMPT, prompts[i], history[i])
          for i in active
      ]
      # Chunk to bound concurrent cache use.
      texts: List[str] = []
      chunk = 16
      for s in range(0, len(batch_prompts), chunk):
        out = sampler(
            input_strings=batch_prompts[s : s + chunk],
            max_generation_steps=max_new_tokens,
            max_prompt_length=cache_size - max_new_tokens - 4,
            echo=False,
            eos_tokens=eos_tokens,
            temperature=0.0,
            seed=0,
        )
        texts.extend(out.text)
      for i, completion in zip(active, texts):
        program = extract_program(completion)
        result = micropython.run(program, max_steps=_EVAL_MAX_STEPS)
        solved = bool(result.ok and result.stdout == golds[i])
        rounds_used[i] += 1
        final_program[i] = program
        final_output[i] = result.stdout
        if round_idx == 0:
          first_solved[i] = solved
        if solved:
          best_solved[i] = True
          done[i] = True
        else:
          history[i].append((program, _format_feedback(result, program)))

  rows = [
      MultiTurnTaskRow(
          task_id=t.id,
          tier=t.tier,
          first_solved=first_solved[i],
          best_solved=best_solved[i],
          rounds_used=rounds_used[i],
          final_program=final_program[i],
          final_output=final_output[i],
      )
      for i, t in enumerate(tasks)
  ]
  return MultiTurnEvalResult(rows=rows)


# ---------------------------------------------------------------------------
# pass@k: the exploration-gap instrument (issue #8 / RL_HEADROOM.md R3).
# ---------------------------------------------------------------------------
#
# Greedy first-attempt solve (evaluate_tasks_multiturn) hides the quantity RL
# actually moves: RL operates at temperature>0, so the relevant baseline is the
# SAMPLED first-attempt pass@1, which can sit far below the argmax. If sampled
# pass@1 << pass@k the correct program is in the policy's support but unreliable
# -- the regime where RLVR earns its keep (it concentrates mass onto modes the
# base/SFT model already samples; cf. RL_HEADROOM.md). If pass@1 ~= pass@k there
# is no exploration gap and RL cannot help, whatever the difficulty. This is the
# diagnostic that decides whether any task refinement is worth running.


@dataclasses.dataclass
class PassKTaskRow:
  """pass@k sampling outcome for one task: ``n_correct`` of ``k`` first attempts."""

  task_id: str
  tier: int
  n_correct: int
  k: int


@dataclasses.dataclass
class PassKResult:
  """Aggregate first-attempt pass@k with the unbiased Chen et al. estimator."""

  rows: List["PassKTaskRow"]
  k: int
  temperature: float

  @property
  def total(self) -> int:
    return len(self.rows)

  @staticmethod
  def _pass_at_m_one(k: int, c: int, m: int) -> float:
    """Unbiased P(at least one of m draws solves) for a task with c/k correct."""
    from math import comb

    if m > k:
      m = k
    if k - c < m:  # fewer than m wrong draws => every m-subset hits a correct one
      return 1.0
    return 1.0 - comb(k - c, m) / comb(k, m)

  def pass_at(self, m: int) -> float:
    if not self.rows:
      return 0.0
    return float(np.mean([self._pass_at_m_one(r.k, r.n_correct, m) for r in self.rows]))

  def per_tier_pass_at(self, m: int) -> Dict[int, Tuple[float, int]]:
    """tier -> (pass@m, n_tasks)."""
    by_tier: Dict[int, List[float]] = {}
    for r in self.rows:
      by_tier.setdefault(r.tier, []).append(self._pass_at_m_one(r.k, r.n_correct, m))
    return {t: (float(np.mean(v)), len(v)) for t, v in by_tier.items()}

  def summary(self, ms: Tuple[int, ...] = (1, 2, 4, 8, 16)) -> str:
    ms = tuple(m for m in ms if m <= self.k) or (1, self.k)
    head = "  ".join(f"pass@{m}={self.pass_at(m):.3f}" for m in ms)
    lines = [f"k={self.k} temp={self.temperature} ({self.total} tasks): {head}"]
    for tier in sorted({r.tier for r in self.rows}):
      cells = []
      for m in ms:
        p, n = self.per_tier_pass_at(m)[tier]
        cells.append(f"@{m}={p:.3f}")
      n = self.per_tier_pass_at(ms[0])[tier][1]
      lines.append(f"  tier {tier} (n={n}): " + " ".join(cells))
    return "\n".join(lines)


def evaluate_passk(
    model,
    tokenizer,
    tasks,
    *,
    k: int = 16,
    max_new_tokens: int = 192,
    max_prompt_length: int = 512,
    temperature: float = 1.0,
    cache_size: int | None = None,
    mesh=None,
    seed: int = 0,
) -> PassKResult:
  """First-attempt pass@k: sample ``k`` round-1 programs per task and grade each.

  Uses the SAME round-1 prompt as the rollout/greedy eval (few-shot + the
  hint-stripped task), sampling at ``temperature`` with a distinct seed per draw
  so the ``k`` samples differ. Returns per-task ``n_correct``/``k``; the unbiased
  estimator on :class:`PassKResult` then gives the pass@1..pass@k curve. Purely a
  measurement -- no training, no multi-turn iteration.
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
  sampler = sampler_lib.Sampler(
      transformer=model, tokenizer=tokenizer, cache_config=cache_config
  )
  eos_tokens = sorted(set([DELPHI_EOS_ID]) | set(program_terminal_eos_tokens(tokenizer)))

  prompts = [build_agent_prompt(strip_answer_hint(t.prompt, t.answer)) for t in tasks]
  golds = [t.answer for t in tasks]
  n_correct = [0] * len(tasks)

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
            top_p=1.0,  # without top_p the tunix Sampler is GREEDY (ignores temp/seed)
            seed=seed + draw,
        )
        texts.extend(out.text)
      for i, completion in enumerate(texts):
        result = micropython.run(extract_program(completion), max_steps=_EVAL_MAX_STEPS)
        if result.ok and result.stdout == golds[i]:
          n_correct[i] += 1

  rows = [
      PassKTaskRow(task_id=t.id, tier=t.tier, n_correct=n_correct[i], k=k)
      for i, t in enumerate(tasks)
  ]
  return PassKResult(rows=rows, k=k, temperature=temperature)


def evaluate_repair_passk(
    model,
    tokenizer,
    *,
    tiers: Tuple[int, ...],
    n_tasks: int = 24,
    k: int = 16,
    max_new_tokens: int = 192,
    max_prompt_length: int = 1280,
    temperature: float = 1.0,
    cache_size: int | None = None,
    mesh=None,
    seed: int = 0,
) -> Tuple["PassKResult", "PassKResult"]:
  """Repair vs synthesis pass@k on the SAME instances (issue #8 / RL_HEADROOM.md R5).

  Synthesis-from-scratch hits the empty-program wall held-out (pass@k ~= pass@1).
  Repair scaffolds the model PAST that wall: it is shown a nearly-correct buggy
  program + its wrong output and must emit a fix -- the feedback-conditioned
  "which edit for which error" behavior that is closest to the CALC tool-result
  copy (issue #5), the one place RL was essential. The question is whether that
  scaffold unlocks ability the from-scratch policy cannot sample, and whether
  fixing is rare-but-present (repair pass@1 << repair pass@k) -- the only signature
  that would give Dr.GRPO a gradient on this model. We measure BOTH on the same
  sampled family instances so the comparison is within-instance.

  Returns ``(synth, repair)`` :class:`PassKResult`. The repair prompt is exactly
  round-2 of the rollout with round-1 seeded by the GIVEN buggy program (via
  :func:`_build_round_prompt`), so it matches the multi-turn fix format the model
  saw in SFT/few-shot.
  """
  import contextlib

  from tunix.generate import sampler as sampler_lib

  rng = random.Random(seed)
  insts: List[Tuple[str, str, str, str]] = []  # (clean_prompt, gold, buggy_src, feedback)
  attempts = 0
  while len(insts) < n_tasks and attempts < n_tasks * 50:
    attempts += 1
    prompt, solution, gold = sample_task(rng, tiers)
    bug = _mutate_to_bug(rng, solution, gold)
    if bug is None:
      continue
    insts.append((strip_answer_hint(prompt, gold), gold, bug[0], bug[1]))

  if cache_size is None:
    cache_size = max_prompt_length + max_new_tokens + 16
  cache_config = sampler_lib.CacheConfig(
      cache_size=cache_size,
      num_layers=model.config.num_layers,
      num_kv_heads=model.config.num_kv_heads,
      head_dim=model.config.head_dim,
  )
  sampler = sampler_lib.Sampler(
      transformer=model, tokenizer=tokenizer, cache_config=cache_config
  )
  eos_tokens = sorted(set([DELPHI_EOS_ID]) | set(program_terminal_eos_tokens(tokenizer)))

  golds = [g for (_p, g, _b, _f) in insts]
  synth_prompts = [build_agent_prompt(p) for (p, _g, _b, _f) in insts]
  repair_prompts = [
      _build_round_prompt(CODE_AGENT_SYSTEM_PROMPT, p, [(b, f)])
      for (p, _g, b, f) in insts
  ]
  synth_correct = [0] * len(insts)
  repair_correct = [0] * len(insts)

  def _sample(prompts: List[str], draw_seed: int) -> List[str]:
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
          top_p=1.0,  # without top_p the tunix Sampler is GREEDY (ignores temp/seed)
          seed=draw_seed,
      )
      texts.extend(out.text)
    return texts

  ctx = mesh if mesh is not None else contextlib.nullcontext()
  with ctx:
    for draw in range(k):
      for prompts, counts in ((synth_prompts, synth_correct), (repair_prompts, repair_correct)):
        for i, completion in enumerate(_sample(prompts, seed + draw)):
          result = micropython.run(extract_program(completion), max_steps=_EVAL_MAX_STEPS)
          if result.ok and result.stdout == golds[i]:
            counts[i] += 1

  tier = tiers[-1] if tiers else 0
  synth = PassKResult(
      rows=[PassKTaskRow(f"synth_{i}", tier, synth_correct[i], k) for i in range(len(insts))],
      k=k,
      temperature=temperature,
  )
  repair = PassKResult(
      rows=[PassKTaskRow(f"repair_{i}", tier, repair_correct[i], k) for i in range(len(insts))],
      k=k,
      temperature=temperature,
  )
  return synth, repair

