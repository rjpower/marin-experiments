"""T0 tool-calling GRPO components for Delphi on tunix (single calculator call).

Milestone **T0**: 2-digit x 2-digit multiplication via a single calculator tool
call. The episode is two turns (``max_steps=2``):

  * turn 1 -- the model emits ``CALC(A * B)``; the :class:`CalcTextToolParser`
    turns that into a calculator :class:`ToolCall`, the env executes the stock
    :class:`CalculatorTool` and injects ``A*B`` back as a ``role=="tool"``
    message.
  * turn 2 -- the model reads the tool result and emits the final numeric
    answer (a plain string), which the env treats as a ``finish`` and scores.

We use the minimal ``CALC(a * b)`` surface rather than Qwen ``<tool_call>`` JSON:
a 447M base LM emits the CALC line cleanly but almost never closes valid JSON, so
on a TPU run the JSON surface measured ``tool_call_rate~=0`` / ``solve_ratio~=0``
(operands mostly right, ``op``/braces mangled, so the tool never executed). CALC
has nothing to balance, so the only hard part is operand-copy -- the skill we
train.

This module provides the genuinely-new T0 pieces (everything else -- the Delphi
loader, mesh, ``RLCluster``, ``GRPOConfig``, ``DelphiRawTextChatParser``,
``_NormalizingGRPOLearner`` -- is reused from the M-port harness):

  * :data:`T0_SYSTEM_PROMPT` -- few-shot tool-use transcripts that make the
    Delphi base LM emit a ``CALC(a * b)`` call from cold; what is learnable is
    OPERAND-COPY (``arg_acc~=0.10`` cold).
  * :func:`build_t0_dataset` -- a grain dataset with ``prompts`` / ``a`` / ``b``
    / ``answer`` columns (M-port grain shape), so ``a``, ``b``, ``answer`` reach
    the learner reward fns as kwargs (they flow through the trajectory's
    ``original_input`` == the env task dict; verified against
    ``agentic_grpo_learner._process_results``).
  * :class:`CalcTextToolParser` -- parses ``CALC(a * b)`` into a calculator
    :class:`ToolCall` (duck-typing the stock parser interface), with
    ``get_tool_prompt`` suppressed (a base LM must NOT see tool docs; the
    few-shot demos carry the format instead).
  * :class:`DelphiToolAgent` -- a thin :class:`ToolAgent` subclass that (a) uses
    the CALC parser and (b) renders the INITIAL task observation (the
    dataset row, keyed ``prompts``) as the user turn -- the stock
    ``ToolAgent._observation_to_messages`` only handles ``tool_outputs`` /
    ``question``, so without this the task text would never reach the model.
  * :class:`CalcToolEnvironment` -- a :class:`ToolEnvironment` subclass that
    stashes the executed calculator result and gives a per-turn, COPY-AWARE
    terminal reward (copy term +0.4 if the final answer copies the tool result,
    solve term +1.0 if it contains the gold product). The dense copy term is what
    lets the base LM's answer be *grounded* in the tool output instead of guessed.
  * :func:`arg_reward` -- the learner reward fn scored on the turn-1 CALC-call
    text (``completions`` == first assistant message): +0.5 for correct operands
    (the KEY turn-1 learnable signal). :func:`format_reward` is also defined (for
    the M-format probe) but T0 no longer sums it -- the lenient closing-optional
    parse plus the dense copy term carry the signal.
  * :func:`t0_metric_fn` -- emits ``tool/tool_call_rate``, ``tool/arg_acc`` and
    ``arithmetic/solve_ratio`` for the dashboards / metric capture.
"""

from __future__ import annotations

import dataclasses
import itertools
import os
import random
import re
import threading
from typing import Any, Dict, List

import grain.python as grain
import jax
import numpy as np
from tunix.rl.agentic.agents.tool_agent import ToolAgent
from tunix.rl.agentic.environments.tool_environment import ToolEnvironment
from tunix.rl.agentic.tools import base_tool
from tunix.rl.agentic.tools.calculator_tool import CalculatorTool

# Structured tool call the stock ToolAgent expects back from a parser's
# ``parse()`` (``ToolCall(name=..., arguments=...)``). Same type the Qwen parser
# returns; we produce it from our ``CALC(a * b)`` surface instead of Qwen JSON.
ToolCall = base_tool.ToolCall


# Temporary diagnostic: when DELPHI_T0_DEBUG=1, t0_metric_fn prints a few raw
# turn-1 completions per step (to inspect the exact CALC text the rollout emits).
_T0_DEBUG_SAMPLES = os.environ.get("DELPHI_T0_DEBUG", "0") == "1"


# Calculator tool registration name (the CALC parser's ToolCall name and the env
# tool map key). The stock CalculatorTool takes {"a", "b", "op"} (op in {+,-,*,/}).
CALCULATOR_TOOL_NAME = "calculator"

# T0 tool map for ToolEnvironment / DelphiToolAgent (name -> tool class).
T0_TOOL_MAP: Dict[str, type] = {CALCULATOR_TOOL_NAME: CalculatorTool}


def install_per_call_rollout_seed(rl_cluster, base_seed: int = 0) -> None:
  """Patches an ``RLCluster``'s rollout to use a FRESH seed on every generate call.

  CRITICAL tunix-0.1.7 gotcha (verified): the agentic learner generates each of
  the ``num_generations`` group members with a SEPARATE
  ``rollout.generate(prompt, rollout_config)`` call, but ``RolloutConfig.seed`` is
  fixed (default ``None`` -> the sampler uses ``PRNGKey(0)`` every call). With the
  SAME prompt + SAME seed, all G generations are BYTE-IDENTICAL -> zero
  intra-group variance -> the group-relative advantage is identically 0 -> GRPO
  has NO gradient (the rollout is degenerate even though each completion is
  individually fine). This is invisible for greedy/eval but fatal for training.

  This installs a thread-safe atomic counter on the cluster's rollout object and
  wraps its ``generate`` so each call folds the next counter value into a fresh
  ``PRNGKey`` (the cluster_config is frozen, so we cannot mutate the seed there;
  we override the bound method on the rollout instance instead). Concurrent
  rollouts (``max_concurrency`` threads) each get a distinct seed, restoring the
  sampling variance GRPO needs WITHOUT serializing the rollouts.

  Args:
    rl_cluster: the constructed ``RLCluster`` whose rollout to patch.
    base_seed: base offset for the per-call seeds (for reproducibility across
      runs while still varying within a run).
  """
  rollout = rl_cluster.rollout
  counter = itertools.count(base_seed)
  lock = threading.Lock()
  original_generate = rollout.generate

  def generate_with_fresh_seed(prompts, rollout_config, **kwargs):
    with lock:
      n = next(counter)
    fresh_config = dataclasses.replace(
        rollout_config, seed=jax.random.PRNGKey(n)
    )
    return original_generate(prompts, fresh_config, **kwargs)

  rollout.generate = generate_with_fresh_seed


def _extract_calculator_result(tool_output: Any) -> str:
  """Extracts the clean numeric result from a calculator tool-output string.

  The env passes the tool output as ``str(ToolOutput)`` (the calculator returns
  ``str(a*b)``, e.g. ``"3840"``, possibly as a float string ``"3840.0"`` if an
  operand parsed as a float). We strip a trailing ``".0"`` so the injected
  ``"Tool result: 3840"`` matches the integer-result few-shot demos and the
  base LM continues with a clean integer on turn 2.

  Args:
    tool_output: the raw tool-output value (string or stringifiable).

  Returns:
    The cleaned result string.
  """
  text = "" if tool_output is None else str(tool_output)
  text = text.strip()
  if text.endswith(".0"):
    text = text[:-2]
  return text


# Few-shot tool-use transcripts. A Delphi base LM has no chat template, so -- as
# for the arithmetic few-shot prefixes -- the demonstrations ARE what make it
# emit the format. We use a deliberately MINIMAL tool surface, ``CALC(a * b)``,
# rather than Qwen ``<tool_call>`` JSON: a 447M base LM emits the CALC line
# cleanly but almost never closes a valid JSON call (a TPU run with the JSON
# surface measured tool_call_rate~=0.00 and solve_ratio~=0.00 -- the model got
# operands mostly right but mangled the ``op`` field / closing braces, so the
# tool never executed). CALC has no quotes, ``op`` field, or braces to balance,
# so the only hard part is copying the operands -- exactly the skill we want to
# train. Each demo is a full 2-turn transcript rendered the way
# DelphiRawTextChatParser renders the live episode: task line, CALC call,
# "Tool result: N" line, final numeric answer.
T0_SYSTEM_PROMPT = (
    "You are a calculator-using assistant. For each problem, write ONE "
    "calculator call CALC(a * b) to compute the product, then read the tool "
    "result and give the final number.\n"
    "Q: 12 * 13\n"
    "CALC(12 * 13)\n"
    "Tool result: 156\n"
    "156\n"
    "Q: 24 * 31\n"
    "CALC(24 * 31)\n"
    "Tool result: 744\n"
    "744\n"
    "Q: 58 * 46\n"
    "CALC(58 * 46)\n"
    "Tool result: 2668\n"
    "2668"
)


# T1 (two chained calls): the second CALC carries the FIRST tool result as an
# argument, so the demos show the model copying the intermediate forward.
T1_SYSTEM_PROMPT = (
    "You are a calculator-using assistant. Use the calculator ONE step at a "
    "time: write CALC(...) for the first product, read the tool result, then "
    "use that result in the next CALC(...), and finally give the number.\n"
    "Q: 12 * 13 * 14\n"
    "CALC(12 * 13)\n"
    "Tool result: 156\n"
    "CALC(156 * 14)\n"
    "Tool result: 2184\n"
    "2184\n"
    "Q: 24 * 31 * 17\n"
    "CALC(24 * 31)\n"
    "Tool result: 744\n"
    "CALC(744 * 17)\n"
    "Tool result: 12648\n"
    "12648"
)


# T2 (three chained calls): a deeper chain a * b * c * d -- the demos show TWO
# intermediate copies forward (turn-1 result into call 2, turn-2 result into
# call 3) before the final copy.
T2_SYSTEM_PROMPT = (
    "You are a calculator-using assistant. Use the calculator ONE step at a "
    "time: write CALC(...) for the first product, read the tool result, use it "
    "in the next CALC(...), keep going until all numbers are multiplied, then "
    "give the final number.\n"
    "Q: 12 * 13 * 14 * 15\n"
    "CALC(12 * 13)\n"
    "Tool result: 156\n"
    "CALC(156 * 14)\n"
    "Tool result: 2184\n"
    "CALC(2184 * 15)\n"
    "Tool result: 32760\n"
    "32760\n"
    "Q: 24 * 31 * 17 * 13\n"
    "CALC(24 * 31)\n"
    "Tool result: 744\n"
    "CALC(744 * 17)\n"
    "Tool result: 12648\n"
    "CALC(12648 * 13)\n"
    "Tool result: 164424\n"
    "164424"
)


# A genuine end-of-line for a T0 turn ends in a digit (the final numeric answer)
# or ``')'`` (the close of a ``CALC(a * b)`` call). A trailing-newline BPE token
# is a valid turn stop ONLY if the char before the newline is one of these (or
# the token is a bare ``"\n"``). With the CALC surface there is no mid-call
# newline hazard (CALC has no quotes/colons/braces to break across lines, unlike
# the Qwen JSON surface that this guarded against), but the digit/``)`` rule is
# still the correct, minimal end-of-line predicate.
_VALID_PRE_NEWLINE = set("0123456789)")


def newline_terminal_eos_tokens(tokenizer) -> list[int]:
  """Returns the token ids that mark a genuine END-OF-LINE for a T0 turn.

  A base LM never emits real EOS, so each single-line T0 turn (the ``CALC(a * b)``
  call line, the final-answer line) must stop at its line break. One BPE fact on
  Delphi's Llama-3 vocab complicates the naive "stop on id 198 (``\\n``)" rule:
  the newline FUSES with the preceding char into one token, so a genuine line end
  is ``")\\n"`` (closing a CALC call) or ``"<digit>\\n"`` (the final answer), NOT
  bare id 198. A 198-only stop misses these, so the model runs past the line.

  So we stop on a token ending in a newline whenever the character immediately
  before that newline is a digit or ``')'`` (a genuine end-of-line for our two
  turn types), plus the bare ``"\\n"`` (id 198) for safety. (The CALC surface has
  no mid-call newline hazard -- no quotes/colons/braces to break across lines --
  so unlike the earlier Qwen-JSON surface there is nothing to exclude here.)

  Args:
    tokenizer: the HF tokenizer (must support ``decode``).

  Returns:
    A sorted list of token ids that mark a genuine end-of-line for a T0 turn.
  """
  eos = []
  for tid in range(int(tokenizer.vocab_size)):
    text = tokenizer.decode([tid])
    if not text.endswith("\n") or "\n" in text[:-1]:
      continue
    before = text[:-1]
    # Bare newline, or a newline fused onto a digit / '>' (genuine line ends).
    if before == "" or before[-1] in _VALID_PRE_NEWLINE:
      eos.append(tid)
  return sorted(eos)


def render_tool_result(content: str) -> str:
  """Renders a tool-output string the way the few-shot demos present it.

  The env (via ``DelphiToolAgent._observation_to_messages``) injects the tool
  output as a ``role=="tool"`` message whose content is the RAW result number;
  ``DelphiRawTextChatParser`` then renders it as ``"Tool result: <content>"`` --
  matching the few-shot ``"Tool result: 156"`` lines so turn-2 generation
  continues with a clean numeric answer.

  Args:
    content: the raw tool-output string (e.g. ``"156"``).

  Returns:
    The content unchanged (the ``"Tool result: "`` prefix is added by the chat
    parser's ``tool`` branch, not here).
  """
  return content


# ---------------------------------------------------------------------------
# Parsing helpers (operand extraction from the turn-1 CALC(a * b) call text).
# ---------------------------------------------------------------------------

_FIRST_INT_RE = re.compile(r"-?\d+")
# The operand body of a ``CALC(a * b`` call, closing ``)`` OPTIONAL. We match on
# the prefix (not requiring the ``)``) because the rollout records the assistant
# turn with its stop token stripped, and on Delphi's Llama-3 BPE the genuine line
# end ``")\n"`` is one fused token -- so a correctly-emitted ``CALC(a * b)`` is
# recorded as ``CALC(a * b`` (confirmed on TPU). The two integer operands survive
# and are unambiguous, so this prefix is the executable / scoreable unit for the
# parser, the operand-copy reward, the format reward, and the metrics alike.
_CALC_PREFIX_RE = re.compile(r"CALC\(\s*(\d+)\s*\*\s*(\d+)")


def parse_tool_call_operands(text: str) -> tuple[int, int] | None:
  """Extracts ``(a, b)`` operands from a ``CALC(a * b)`` call in ``text`` (lenient).

  Used by the operand-copy reward / metric (the KEY learnable T0 signal), so it
  is deliberately LENIENT: it matches ``CALC(a * b`` with the closing ``)``
  optional, so an unclosed cold call still yields its operand-copy gradient.
  Returns ``None`` only when no ``CALC(`` with both operands is present.

  Args:
    text: the raw turn-1 assistant text (the tool call).

  Returns:
    ``(a, b)`` as ints, or ``None`` if no operands can be recovered.
  """
  match = _CALC_PREFIX_RE.search(text)
  if match is None:
    return None
  try:
    return int(match.group(1)), int(match.group(2))
  except ValueError:
    return None


def is_well_formed_tool_call(text: str) -> bool:
  """Returns True iff ``text`` contains an executable ``CALC(a * b)`` call.

  Closing-``)``-OPTIONAL by design. The rollout records the assistant turn with
  its stop token STRIPPED, and on Delphi's Llama-3 BPE the genuine line end
  ``")\\n"`` is a SINGLE fused token -- so when the model correctly emits
  ``CALC(a * b)`` and stops, the ``")\\n"`` token (and with it the ``)``) is
  removed, leaving ``CALC(a * b`` in the recorded text (confirmed on TPU: every
  rollout completion was ``'CALC(64 * 64'`` etc., never with a trailing ``)``).
  The two integer operands survive intact and are unambiguous, so we treat a call
  with both operands present as executable regardless of the (often-stripped)
  closing ``)``. This is what makes the tool actually run; ``tool_call_rate``
  measures it, and :class:`CalcTextToolParser` executes on the same predicate.
  """
  return _CALC_PREFIX_RE.search(text) is not None


def _coerce_str(value: Any) -> str:
  """Collapses a numpy/bytes/str leaf to a plain Python str (reward kwargs)."""
  if isinstance(value, str):
    return value
  arr = np.asarray(value)
  if arr.ndim == 0:
    item = arr.item()
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)
  return str(value)


def _coerce_int(value: Any) -> int:
  """Collapses a numpy/str leaf to a plain Python int (a/b/answer kwargs)."""
  return int(_FIRST_INT_RE.search(_coerce_str(value)).group())


# ---------------------------------------------------------------------------
# Dataset.
# ---------------------------------------------------------------------------


def _make_t0_problem(rng: random.Random) -> tuple[str, int, int, str]:
  """Generates one 2-digit x 2-digit multiply problem.

  Both operands are drawn from ``[11, 99]`` so the product is wide enough that
  Delphi's internal multiply is ~0% (forcing genuine tool use) while the result
  stays short enough to copy reliably.

  Args:
    rng: a seeded ``random.Random`` for reproducibility.

  Returns:
    ``(prompt, a, b, answer)`` where ``prompt`` is the user-turn text
    (``"Q: a * b"``), ``a`` / ``b`` are the operands and ``answer`` is
    ``str(a*b)``.
  """
  a = rng.randint(11, 99)
  b = rng.randint(11, 99)
  return f"Q: {a} * {b}", a, b, str(a * b)


class _T0Source(grain.RandomAccessDataSource):
  """A grain source of ``(prompt, a, b, answer)`` T0 rows (deterministic)."""

  def __init__(self, n: int, seed: int):
    rng = random.Random(seed)
    self._rows = [_make_t0_problem(rng) for _ in range(n)]

  def __len__(self) -> int:
    return len(self._rows)

  def __getitem__(self, idx: int) -> tuple[str, int, int, str]:
    return self._rows[idx]


def build_t0_dataset(n: int, seed: int, batch_size: int) -> grain.MapDataset:
  """Builds a batched grain dataset of T0 multiply problems for GRPO.

  Emits rows with ``prompts`` (user-turn strings), ``a`` / ``b`` (operands) and
  ``answer`` (gold product string) columns, using the M-port grain shape so each
  batched column is a single numpy-array leaf (HF ``datasets.batch`` instead
  yields per-column Python lists, which tunix's ``tree_map`` corrupts). The
  non-``prompts`` columns flow to the learner reward fns / metric fn as kwargs
  via the trajectory's ``original_input`` (== the env task dict).

  Args:
    n: number of distinct problems to generate.
    seed: PRNG seed for the problem set.
    batch_size: prompts per global step.

  Returns:
    A batched ``grain.MapDataset`` with ``prompts`` / ``a`` / ``b`` / ``answer``
    columns.
  """
  source = _T0Source(n, seed)

  def _to_columns(batch):
    # grain's .batch collates the 4-tuple source rows field-wise into a 4-tuple
    # of numpy arrays (each a single leaf).
    prompts, a, b, answers = batch
    return {"prompts": prompts, "a": a, "b": b, "answer": answers}

  return grain.MapDataset.source(source).batch(batch_size).map(_to_columns)


def _make_t1_problem(rng: random.Random) -> tuple[str, int, int, int, str]:
  """Generates one CHAINED ``a * b * c`` problem (T1: two dependent calls).

  The agent must compute ``a*b`` (turn 1), COPY that ~4-digit intermediate into
  a second ``CALC(<a*b> * c)`` (turn 2), then copy the final product (turn 3) --
  so the turn-2 call exercises true chaining (using turn-1's tool output as an
  argument), not just two independent calls. All operands are 2-digit ``[11,99]``.

  Returns:
    ``(prompt, a, b, c, answer)`` where ``prompt`` is ``"Q: a * b * c"`` and
    ``answer`` is ``str(a*b*c)``.
  """
  a = rng.randint(11, 99)
  b = rng.randint(11, 99)
  c = rng.randint(11, 99)
  return f"Q: {a} * {b} * {c}", a, b, c, str(a * b * c)


class _T1Source(grain.RandomAccessDataSource):
  """A grain source of ``(prompt, a, b, c, answer)`` T1 rows (deterministic)."""

  def __init__(self, n: int, seed: int):
    rng = random.Random(seed)
    self._rows = [_make_t1_problem(rng) for _ in range(n)]

  def __len__(self) -> int:
    return len(self._rows)

  def __getitem__(self, idx: int) -> tuple[str, int, int, int, str]:
    return self._rows[idx]


def build_t1_dataset(n: int, seed: int, batch_size: int) -> grain.MapDataset:
  """Builds a batched grain dataset of chained ``a * b * c`` problems for GRPO.

  Emits ``prompts`` / ``a`` / ``b`` / ``c`` / ``answer`` columns. ``a`` / ``b``
  feed the turn-1 :func:`arg_reward` / :func:`t0_metric_fn` (the first call is
  ``CALC(a * b)``); ``c`` rides along as an unused reward kwarg; ``answer``
  (``str(a*b*c)``) is the gold the :class:`CalcToolEnvironment` scores against.
  """
  source = _T1Source(n, seed)

  def _to_columns(batch):
    prompts, a, b, c, answers = batch
    return {"prompts": prompts, "a": a, "b": b, "c": c, "answer": answers}

  return grain.MapDataset.source(source).batch(batch_size).map(_to_columns)


def _make_t2_problem(rng: random.Random) -> tuple[str, int, int, int, int, str]:
  """Generates one CHAINED ``a * b * c * d`` problem (T2: three dependent calls).

  The agent must compute ``a*b`` (turn 1), COPY it into ``CALC(<a*b> * c)``
  (turn 2), COPY that ~6-digit intermediate into ``CALC(<a*b*c> * d)`` (turn 3),
  then copy the final ~8-digit product -- a deeper chain than T1. All operands
  are 2-digit ``[11,99]``.

  Returns:
    ``(prompt, a, b, c, d, answer)`` where ``prompt`` is ``"Q: a * b * c * d"``
    and ``answer`` is ``str(a*b*c*d)``.
  """
  a = rng.randint(11, 99)
  b = rng.randint(11, 99)
  c = rng.randint(11, 99)
  d = rng.randint(11, 99)
  return f"Q: {a} * {b} * {c} * {d}", a, b, c, d, str(a * b * c * d)


class _T2Source(grain.RandomAccessDataSource):
  """A grain source of ``(prompt, a, b, c, d, answer)`` T2 rows (deterministic)."""

  def __init__(self, n: int, seed: int):
    rng = random.Random(seed)
    self._rows = [_make_t2_problem(rng) for _ in range(n)]

  def __len__(self) -> int:
    return len(self._rows)

  def __getitem__(self, idx: int) -> tuple[str, int, int, int, int, str]:
    return self._rows[idx]


def build_t2_dataset(n: int, seed: int, batch_size: int) -> grain.MapDataset:
  """Builds a batched grain dataset of chained ``a * b * c * d`` problems.

  Emits ``prompts`` / ``a`` / ``b`` / ``c`` / ``d`` / ``answer`` columns. ``a`` /
  ``b`` feed the turn-1 :func:`arg_reward` / :func:`t0_metric_fn` (the first call
  is ``CALC(a * b)``); ``c`` / ``d`` ride along as unused reward kwargs;
  ``answer`` (``str(a*b*c*d)``) is the gold the env scores against.
  """
  source = _T2Source(n, seed)

  def _to_columns(batch):
    prompts, a, b, c, d, answers = batch
    return {"prompts": prompts, "a": a, "b": b, "c": c, "d": d, "answer": answers}

  return grain.MapDataset.source(source).batch(batch_size).map(_to_columns)


# ---------------------------------------------------------------------------
# CALC text parser + Delphi tool agent.
# ---------------------------------------------------------------------------


class CalcTextToolParser:
  """Parses the ``CALC(a * b)`` surface into a calculator :class:`ToolCall`.

  Duck-types the stock tool-parser interface that :class:`ToolAgent` uses:

    * ``parse(model_response) -> list[ToolCall]`` -- returns a single calculator
      call when a ``CALC(a * b`` with both operands is present (closing ``)``
      OPTIONAL), else ``[]``. An empty list makes ``ToolAgent.update_from_model``
      treat the response as the ``finish`` / final answer (what we want on turn 2,
      where the model emits the bare product). The ``)`` is optional because the
      rollout strips the stop token, and on this BPE the closing ``")\\n"`` is one
      fused token -- so a correctly-closed ``CALC(a * b)`` is recorded as
      ``CALC(a * b`` (confirmed on TPU). The two operands survive and are
      unambiguous, so we execute on them.
    * ``get_tool_prompt(...) -> ""`` -- suppressed: a base LM must not see tool
      prose; the few-shot :data:`T0_SYSTEM_PROMPT` carries the format.

  This replaces the Qwen ``<tool_call>`` JSON parser: a 447M base LM emits the
  CALC line cleanly but almost never closes valid JSON (measured on TPU), so the
  minimal CALC surface is what makes the tool actually execute.
  """

  def get_tool_prompt(self, tools=None, schema_style: str = "openai") -> str:
    del tools, schema_style
    return ""

  def parse(self, model_response: str) -> list:
    operands = parse_tool_call_operands(model_response or "")
    if operands is None:
      return []
    a, b = operands
    return [
        ToolCall(
            name=CALCULATOR_TOOL_NAME,
            arguments={"a": a, "b": b, "op": "*"},
        )
    ]


class DelphiToolAgent(ToolAgent):
  """A :class:`ToolAgent` for Delphi: suppressed tool docs + task-as-user-turn.

  Two deviations from the stock ``ToolAgent``:

    1. Uses :class:`CalcTextToolParser` (the ``CALC(a * b)`` surface) so
       ``tools_prompt`` is empty -- no tool prose leaks into the system message;
       only the few-shot :data:`T0_SYSTEM_PROMPT` does.
    2. Overrides ``_observation_to_messages`` to render the INITIAL task
       observation. The env's first observation is the dataset row dict, keyed
       ``prompts`` (e.g. ``"Q: 47 * 53"``). The stock ``ToolAgent`` only handles
       ``tool_outputs`` / ``question``, so the task text would otherwise never
       reach the model. Tool-output observations are still handled by the parent.
  """

  def __init__(self, system_prompt: str):
    # Reproduce ToolAgent.__init__ but with the suppressed parser and the T0
    # tool map (so parse() recognizes the calculator surface). We avoid calling
    # super().__init__ with tool_parser_name since the stock signature would
    # re-instantiate the verbose parser.
    from tunix.rl.agentic.tools import tool_manager as _tool_manager
    from tunix.rl.agentic.agents import base_agent as _base_agent

    self.tool_manager = _tool_manager.ToolManager(tool_map=T0_TOOL_MAP)
    self.tool_parser = CalcTextToolParser()
    self.tools_prompt = self.tool_parser.get_tool_prompt(
        self.tool_manager.get_tools()
    )
    _base_agent.ConversationAgentBase.__init__(self, system_prompt=system_prompt)

  def _observation_to_messages(
      self,
      observation: Any,
      reward: float,
      done: bool,
      info: Dict[str, Any],
  ) -> None:
    del reward, done, info
    if isinstance(observation, dict):
      # Initial task observation: the dataset row dict, keyed "prompts".
      if "prompts" in observation:
        content = observation["prompts"]
        self._messages.append(
            {"role": "user", "content": "" if content is None else str(content)}
        )
        return
      # Tool-output observation: inject the RAW tool result as the tool message
      # content, so DelphiRawTextChatParser renders it as exactly
      # "Tool result: <result>" -- matching the few-shot demos. (The stock
      # ToolAgent prefixes "Tool returned result: ", which would double the
      # prefix to "Tool result: Tool returned result: ..." and diverge from the
      # demos, derailing the base LM's turn-2 continuation.)
      if "tool_outputs" in observation:
        for call_id, output in observation["tool_outputs"].items():
          self._messages.append({
              "role": "tool",
              "tool_call_id": call_id,
              "content": _extract_calculator_result(output),
          })
        return
      # Terminal step: ToolEnvironment returns observation={} when the episode
      # is done (finish/string action). Nothing to inject; return quietly
      # instead of letting ToolAgent log "Unknown dict observation format: {}".
      if not observation:
        return
    # question / string observations: defer to ToolAgent.
    super()._observation_to_messages(observation, 0.0, False, {})


# ---------------------------------------------------------------------------
# Environment with a copy-aware shaped reward.
# ---------------------------------------------------------------------------


class CalcToolEnvironment(ToolEnvironment):
  """``ToolEnvironment`` with a per-turn, copy-aware terminal reward.

  T0 needs the model to learn TWO copy-from-context skills: (1) copy the task
  operands into the ``CALC`` call (turn 1), and (2) copy the executed tool result
  into the final answer (turn 2). A single episode-level "answer == a*b" reward is
  too sparse -- it fires only when BOTH skills succeed at once, and (worse) gives
  the turn-2 tokens no dedicated signal, so the base LM's prior to *compute*
  ``a*b`` itself (badly) is never corrected and the guessed answers just get
  reinforced as noise. On TPU this stalled at ``solve~=0`` with ``arg_acc``
  oscillating.

  This env adds a DENSE turn-2 signal by stashing the executed calculator result
  and rewarding the final answer for COPYING it, decoupled from operand
  correctness:

    * copy term  (+0.4): final answer contains the injected tool result (teaches
      the copy MECHANIC even when the operands -- hence the result -- are wrong).
    * solve term (+1.0): final answer contains the gold product ``a*b`` (the
      goal; only reachable via correct operands AND a correct copy).

  The learner :func:`arg_reward` (+0.5, correct operands) is summed on top. So a
  fully-correct episode scores ``0.5 + 0.4 + 1.0 = 1.9`` and the
  non-solve maximum is ``0.5 + 0.4 = 0.9 < 1.0`` -- keeping the
  ``solve_ratio = (summed reward >= 1.0)`` metric a clean "did the solve term
  fire" indicator.
  """

  def __init__(self, *args, **kwargs):
    self._last_tool_result = None
    # We compute our own shaped reward; ignore any reward_fn passed via
    # env_kwargs and bind the env's _shaped_reward as the trajectory reward.
    kwargs.pop("reward_fn", None)
    super().__init__(*args, reward_fn=self._shaped_reward, **kwargs)

  def _execute_tool_calls(self, action):
    """Executes the tool calls and stashes the calculator's numeric result."""
    outputs = super()._execute_tool_calls(action)
    if isinstance(outputs, dict):
      for output in outputs.values():
        result = _extract_calculator_result(output)
        if result:
          self._last_tool_result = result
    return outputs

  def _shaped_reward(self, task: Dict[str, Any], action: Any) -> float:
    """Copy-aware terminal reward: copy term (+0.4) + solve term (+1.0).

    The gold value is the precomputed ``task["answer"]`` (so this generalizes
    over the whole CALC curriculum: ``a*b`` for T0, ``a*b*c`` for T1's chained
    calls, etc.); it falls back to ``a*b`` when no answer column is present.
    """
    try:
      gold = _coerce_int(task["answer"])
    except (KeyError, AttributeError, ValueError, TypeError):
      try:
        gold = _coerce_int(task["a"]) * _coerce_int(task["b"])
      except (KeyError, AttributeError, ValueError, TypeError):
        return 0.0
    answer_text = action if isinstance(action, str) else str(action)
    reward = 0.0
    # Dense copy term: did the final answer copy the injected tool result?
    result = self._last_tool_result
    if result is not None and re.search(
        rf"(?<!\d){re.escape(result)}(?!\d)", answer_text
    ):
      reward += 0.4
    # Solve term: did the final answer contain the gold value as a standalone
    # integer (so "24" does not spuriously match inside "2491")?
    solved = re.search(rf"(?<!\d){gold}(?!\d)", answer_text) is not None
    if solved:
      reward += 1.0
    if _T0_DEBUG_SAMPLES:
      print(
          f"[t0-dbg] final gold={gold} toolres={result} "
          f"reward={reward} answer={answer_text[:80]!r}",
          flush=True,
      )
    return reward


# ---------------------------------------------------------------------------
# Rewards.
# ---------------------------------------------------------------------------


def arg_reward(prompts, completions, a, b, **kwargs) -> list[float]:
  """+0.5 when the turn-1 tool call carries the correct operands ``{a, b}``.

  This is the KEY learnable T0 signal: cold ``arg_acc~=0.10`` (Delphi parrots the
  last few-shot operands or duplicates the first operand), and GRPO trains it up.
  ``completions`` is the FIRST assistant message (the turn-1 ``CALC(a * b)``
  text) -- confirmed against ``agentic_grpo_learner._process_results`` (it
  extracts ``next(msg for msg in conversation if role=="assistant")``).

  Args:
    prompts: batch of prompt strings (unused).
    completions: batch of turn-1 tool-call texts.
    a: batch of first operands (forwarded dataset column).
    b: batch of second operands (forwarded dataset column).
    **kwargs: other forwarded columns (unused).

  Returns:
    One float per completion: 0.5 if the parsed operands equal ``{a, b}``, else
    0.0.
  """
  del prompts, kwargs
  rewards: list[float] = []
  for completion, ai, bi in zip(completions, a, b):
    operands = parse_tool_call_operands(str(completion))
    want = {_coerce_int(ai), _coerce_int(bi)}
    rewards.append(0.5 if (operands is not None and set(operands) == want) else 0.0)
  return rewards


def format_reward(prompts, completions, **kwargs) -> list[float]:
  """+0.1 when the turn-1 ``completions`` text is a CLOSED well-formed call.

  Uses the STRICT :func:`is_well_formed_tool_call` (a closed ``CALC(a * b)`` with
  the trailing ``)``) so this term rewards the model for emitting a complete,
  executable call -- the format skill GRPO should drive toward. (The
  operand-copy reward :func:`arg_reward` is scored leniently on the call prefix
  so cold unclosed calls still get an operand gradient.)

  Args:
    prompts: batch of prompt strings (unused).
    completions: batch of turn-1 tool-call texts.
    **kwargs: forwarded columns (unused).

  Returns:
    One float per completion: 0.1 if a closed well-formed calculator call is
    present.
  """
  del prompts, kwargs
  return [0.1 if is_well_formed_tool_call(str(c)) else 0.0 for c in completions]


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------


def t0_metric_fn(prompts, completions, rewards, advantages, a, b, **kwargs) -> dict:
  """Reports T0 dashboards: tool_call_rate, arg_acc, solve_ratio.

  * ``tool/tool_call_rate`` -- STRICT: fraction of turn-1 completions that are a
    CLOSED, executable ``CALC(a * b)`` call (:func:`is_well_formed_tool_call`).
    Cold this is low (the base LM often omits the closing ``)``); GRPO drives it
    up via :func:`format_reward`.
  * ``tool/arg_acc`` -- LENIENT: fraction whose operands (recovered from the call
    PREFIX, closed or not) equal the gold ``{a, b}``
    (:func:`parse_tool_call_operands`). This is the KEY learnable operand-copy
    skill (~0.10 cold) and the dense GRPO signal -- visible even on unclosed
    cold calls.
  * ``arithmetic/solve_ratio`` -- fraction whose SUMMED reward >= 1.0, i.e. the
    env answer-in-output reward (+1.0) fired => the final answer contained the
    correct product. Mirrors ``arithmetic.metric_fn``'s solved-threshold
    convention so ``_AgenticMetricsCapture`` can read it.

  Args:
    prompts: batch of prompts (unused).
    completions: batch of turn-1 tool-call texts.
    rewards: per-completion summed rewards (env + learner reward fns).
    advantages: per-completion advantages (unused).
    a: batch of first operands (forwarded column).
    b: batch of second operands (forwarded column).
    **kwargs: other forwarded columns (unused).

  Returns:
    A dict of metric name -> ``(value, aggregation_fn)``.
  """
  del prompts, advantages, kwargs
  n = len(completions)
  # DEBUG (temporary): surface a couple of raw turn-1 completions so we can see
  # exactly what the rollout emits (e.g. whether the closing ')' is present).
  if _T0_DEBUG_SAMPLES:
    for k in range(min(3, n)):
      print(f"[t0-dbg] completion[{k}]={str(completions[k])[:90]!r}", flush=True)
  call_hits = 0  # STRICT: closed, executable CALC(a * b) calls.
  arg_hits = 0   # LENIENT: operands (prefix) match gold {a, b}.
  for completion, ai, bi in zip(completions, a, b):
    text = str(completion)
    if is_well_formed_tool_call(text):
      call_hits += 1
    operands = parse_tool_call_operands(text)
    if operands is not None and set(operands) == {_coerce_int(ai), _coerce_int(bi)}:
      arg_hits += 1
  tool_call_rate = call_hits / n if n else 0.0
  arg_acc = arg_hits / n if n else 0.0

  rewards = np.asarray(rewards, dtype=np.float32)
  solved = rewards >= 1.0
  solve_ratio = float(solved.mean()) if solved.size else 0.0
  return {
      "tool/tool_call_rate": (tool_call_rate, np.mean),
      "tool/arg_acc": (arg_acc, np.mean),
      "arithmetic/solve_ratio": (solve_ratio, np.mean),
  }
