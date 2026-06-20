# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-turn agentic-CODING environment for Delphi (issue #7).

The model is shown a coding task and must emit a short Python program; we run the
program through the purely-functional :mod:`micropython` interpreter and reward an
*exact* stdout match against the gold output. The interpreter is the execution
environment + verifier (the model's one "tool", invoked once per episode by the
grader), mirroring how the CALC stages used a calculator -- but here the model
writes the code rather than calling a fixed tool.

Why PARAMETERIZED task families (not the fixed 50 tasks) for training? With an
exact-stdout reward, a fixed task has a fixed gold, so a model could "solve" it by
emitting the constant (``print(55)``) instead of writing a program. Sampling
random parameters per prompt makes the gold vary, so a constant can't win a GRPO
group -- the model is pushed to write a *general* program that reads the operands
from the prompt and computes. This is the same trick the CALC stages used (random
operands). The fixed 50 tasks in :mod:`coding_tasks` are held out as the EVAL set
(:func:`evaluate_tasks`): "how far did we get" = greedy solve-rate per tier on
those, with their answer-leaking hints stripped (:func:`strip_answer_hint`).

Reusable pieces:
  * :data:`CODE_FEWSHOT` -- the few-shot prompt prefix (3 worked demos: constant,
    arithmetic, loop) that carries the ``Task: ... <program> END`` format to the
    base LM (which has no chat template). Both the RL prompt and the SFT warm-up
    use it, byte-identical (train/RL prompt-match, invariant D in ``AGENTS.md``).
  * :func:`build_code_prompt` / :func:`extract_program` -- prompt builder + the
    completion->program parser (cuts at the ``END`` sentinel / a hallucinated next
    ``Task:``).
  * :func:`code_reward` -- a DENSE Dr.GRPO reward (has_code/ran_ok/output-overlap/
    exact) so the group has non-zero advantage even before any sample is exact.
  * :func:`code_metric_fn` -- solve_ratio (exact) / ran_ok / has_code for logging.
  * :func:`code_segments` -- the execution-format SFT transcript builder (consumed
    by :func:`agentic_sft.run_sft_warmup` with ``prompt_prefix=CODE_FEWSHOT``).
  * :func:`build_code_dataset` -- the grain dataset (``prompts`` + ``answer``).
  * :func:`evaluate_tasks` / :func:`greedy_completions` -- the eval-on-50 harness.
"""

from __future__ import annotations

import dataclasses
import random
import re
from typing import Any, Callable, Dict, List, Tuple

import grain.python as grain
import numpy as np

import environments.micropython as micropython

# Step budgets for the interpreter. Training programs are tiny but a buggy model
# program can loop, so the reward grader caps low (most failures are instant
# syntax/name errors anyway); the eval set has valid bounded solutions so it caps
# higher (e.g. naive fib recursion).
_GRADE_MAX_STEPS = 50_000
_EVAL_MAX_STEPS = 400_000


# --- The few-shot format prefix ------------------------------------------------
#
# Carries the Task/program/END format to the base LM. Ends WITHOUT a trailing
# newline so that build_code_prompt and the SFT prefix (run_sft_warmup appends a
# single "\n") produce byte-identical context (invariant D). Three demos span the
# format's range: a constant print, a one-line expression, and a multi-line loop.
CODE_FEWSHOT = (
    "Write a short Python program that prints the answer for each task, "
    "then a line containing only END.\n"
    "\n"
    "Task: Print the number 7.\n"
    "print(7)\n"
    "END\n"
    "Task: Print the result of 8 times 9.\n"
    "print(8 * 9)\n"
    "END\n"
    "Task: Print the sum of the integers from 1 to 4 (inclusive).\n"
    "total = 0\n"
    "for i in range(1, 5):\n"
    "  total += i\n"
    "print(total)\n"
    "END"
)


def build_code_prompt(task_prompt: str) -> str:
  """Builds the full RL/eval prompt: few-shot prefix + the target task line.

  The model continues by emitting ``<program>\\nEND``. ``CODE_FEWSHOT + "\\n"``
  matches exactly what :func:`agentic_sft.run_sft_warmup` prepends when given
  ``prompt_prefix=CODE_FEWSHOT`` (so the SFT context == the RL prompt).
  """
  return f"{CODE_FEWSHOT}\nTask: {task_prompt}\n"


# --- Completion -> program parsing ---------------------------------------------

# The model's program ends at the END sentinel; defensively also stop at a
# hallucinated next "Task:" / repeated instruction line, or a long blank gap.
_STOP_MARKERS = ("\nEND", "\nTask:", "\nWrite a short", "\n\n\n")


def extract_program(completion: str) -> str:
  """Extracts the Python program from a model completion.

  Cuts the completion at the earliest stop marker (the ``END`` sentinel or a
  hallucinated next task) and trims surrounding newlines, yielding the program
  text to hand to :func:`micropython.run`. Returns ``""`` if nothing precedes the
  first marker (an empty / sentinel-only turn).
  """
  text = completion
  cut = len(text)
  for marker in _STOP_MARKERS:
    idx = text.find(marker)
    if idx != -1:
      cut = min(cut, idx)
  return text[:cut].strip("\n")


# --- Grading -------------------------------------------------------------------


def _output_similarity(out: str, gold: str) -> float:
  """A dense [0, 1] closeness between a program's stdout and the gold output.

  Longest-common-prefix length normalized by the longer string, so matching the
  first few lines (e.g. of a multi-line FizzBuzz) earns partial credit and an
  over-long or diverging output is penalized. 1.0 iff ``out == gold``.
  """
  if out == gold:
    return 1.0
  if not gold and not out:
    return 1.0
  longer = max(len(out), len(gold))
  if longer == 0:
    return 0.0
  k = 0
  limit = min(len(out), len(gold))
  while k < limit and out[k] == gold[k]:
    k += 1
  return k / longer


def _coerce_gold(value: Any) -> str:
  """Coerces a forwarded ``answer`` column element to a plain ``str``."""
  if isinstance(value, str):
    return value
  arr = np.asarray(value)
  return str(arr.item()) if arr.ndim == 0 else str(value)


def grade_program(program: str, gold: str, *, max_steps: int = _GRADE_MAX_STEPS) -> Dict[str, float]:
  """Runs ``program`` and scores it against ``gold`` (the gold stdout).

  Returns the per-component signals used by both the reward and the metric:
  ``has_code`` (emitted a non-empty program), ``ran_ok`` (parsed + ran with no
  error), ``exact`` (stdout exactly == gold), ``partial`` (dense output overlap).
  """
  if not program.strip():
    return {"has_code": 0.0, "ran_ok": 0.0, "exact": 0.0, "partial": 0.0}
  result = micropython.run(program, max_steps=max_steps)
  ran_ok = 1.0 if result.ok else 0.0
  exact = 1.0 if (result.ok and result.stdout == gold) else 0.0
  partial = _output_similarity(result.stdout, gold) if result.ok else 0.0
  return {"has_code": 1.0, "ran_ok": ran_ok, "exact": exact, "partial": partial}


# Dense reward weights. Exact match dominates (cliff at correctness) but
# has_code/ran_ok/partial give a smooth climb so the Dr.GRPO group has non-zero
# advantage even before any sample is exactly right (the cold-start wall).
_W_HAS_CODE = 0.1
_W_RAN_OK = 0.3
_W_PARTIAL = 0.6
_W_EXACT = 1.0


def code_reward(prompts, completions, answer, **kwargs) -> List[float]:
  """Dense per-completion reward for the coding task (summed by the learner).

  ``reward = 0.1*has_code + 0.3*ran_ok + 0.6*partial + 1.0*exact`` -- an exact
  solution scores 2.0; a program that runs but prints the wrong thing scores
  0.4..1.0 by output overlap; non-code scores 0. The dense middle is what lets
  GRPO/Dr.GRPO bootstrap from a base LM that is rarely exact at first.

  Args:
    prompts: batch of prompt strings (unused).
    completions: batch of decoded completion strings.
    answer: forwarded ``answer`` column (gold stdout per row).
    **kwargs: other forwarded columns (unused).
  """
  del prompts, kwargs
  rewards: List[float] = []
  for completion, gold in zip(completions, answer):
    g = _coerce_gold(gold)
    s = grade_program(extract_program(str(completion)), g)
    rewards.append(
        _W_HAS_CODE * s["has_code"]
        + _W_RAN_OK * s["ran_ok"]
        + _W_PARTIAL * s["partial"]
        + _W_EXACT * s["exact"]
    )
  return rewards


def code_metric_fn(prompts, completions, rewards, advantages, answer, **kwargs) -> dict:
  """Reports solve_ratio (exact) / ran_ok / has_code over the batch.

  Recomputes the grade from the completions (independent of the summed reward) so
  ``coding/solve_ratio`` is a clean exact-stdout-match rate.
  """
  del prompts, rewards, advantages, kwargs
  exact, ran, has = [], [], []
  for completion, gold in zip(completions, answer):
    g = _coerce_gold(gold)
    s = grade_program(extract_program(str(completion)), g)
    exact.append(s["exact"])
    ran.append(s["ran_ok"])
    has.append(s["has_code"])
  mean = lambda xs: float(np.mean(xs)) if xs else 0.0
  return {
      "coding/solve_ratio": (mean(exact), np.mean),
      "coding/ran_ok": (mean(ran), np.mean),
      "coding/has_code": (mean(has), np.mean),
  }


# --- Parameterized task families -----------------------------------------------
#
# Each family samples (prompt, solution) from random parameters; the gold is
# computed by running the solution through micropython. Prompts never state the
# answer value, so the model must WRITE code that computes it (a constant can't
# win a group of randomized golds). The families mirror the skills in the fixed
# 50-task eval ladder (coding_tasks.py), tier for tier.

_WORDS = (
    "cat", "dog", "house", "table", "python", "river", "planet", "forest",
    "yellow", "silver", "garden", "rocket", "purple", "orange", "wizard",
    "dragon", "castle", "winter", "summer", "pencil",
)

# Short lowercase words (no apostrophes) for building random sentences in the
# tier-5 families that operate on whitespace-separated text (reverse-words,
# most-common-word, ...). Kept lowercase + single-token so split(' ') / join
# round-trips exactly and the gold stdout is unambiguous.
_PHRASE_WORDS = (
    "the", "quick", "brown", "fox", "lazy", "dog", "red", "blue", "green",
    "sun", "moon", "star", "code", "runs", "fast", "slow", "tree", "bird",
    "rain", "snow",
)

# Short words including several palindromes so the is-palindrome family yields a
# mix of yes/no golds (a constant answer can't win the GRPO group).
_PALINDROME_WORDS = (
    "cat", "dog", "house", "table", "python", "river", "planet", "forest",
    "level", "noon", "radar", "civic", "rotor", "kayak", "madam", "refer",
    "racecar", "deed", "stats", "tenet",
)

_LOWERCASE = "abcdefghijklmnopqrstuvwxyz"

# Multi-word phrases for the literal-print family (no apostrophes -- they would
# break the single-quoted string literal). Disjoint from the eval phrases
# ("Hello, World!" / "the quick brown fox"), so those stay held out.
_PHRASES = (
    "Good morning", "I love Python", "red green blue", "open the door",
    "time to code", "keep it simple", "two plus two", "left and right",
    "up and down", "the answer", "make it work", "hello there friend",
    # Mixed case + punctuation so exact-copy of capitalized/punctuated phrases is
    # in distribution (no apostrophes -- they break the single-quoted literal).
    "Hello there!", "Good Morning!", "Yes, indeed.", "Stop and go.",
    "Up, up, away!", "Red, green, blue.", "Wow, nice!", "Keep going!",
)

_UPPERCASE = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

Sampler = Callable[[random.Random], Tuple[str, str]]


@dataclasses.dataclass(frozen=True)
class Family:
  """A parameterized task family: ``sample(rng) -> (prompt, solution)``."""

  id: str
  tier: int
  sample: Sampler


def _f0_int(rng):
  n = rng.randint(0, 999)
  if rng.random() < 0.2:
    n = -n
  return f"Print the number {n}.", f"print({n})"


def _f0_word(rng):
  w = rng.choice(_WORDS)
  return f"Print the word {w} (lowercase, no quotes).", f"print('{w}')"


_BINOPS = (
    ("plus", "+"), ("minus", "-"), ("times", "*"),
)


def _f1_binop(rng):
  word, op = rng.choice(_BINOPS)
  a, b = rng.randint(0, 99), rng.randint(0, 99)
  return f"Print the result of {a} {word} {b}.", f"print({a} {op} {b})"


def _f1_floordiv(rng):
  b = rng.randint(2, 12)
  a = rng.randint(b, 99)
  return (
      f"Print the integer (floor) division of {a} by {b}.",
      f"print({a} // {b})",
  )


def _f1_mod(rng):
  b = rng.randint(2, 12)
  a = rng.randint(1, 99)
  return f"Print the remainder when {a} is divided by {b}.", f"print({a} % {b})"


def _f1_power(rng):
  a, b = rng.randint(2, 9), rng.randint(2, 5)
  return f"Print {a} raised to the power {b}.", f"print({a} ** {b})"


def _f1_parens(rng):
  a, b, c = rng.randint(1, 20), rng.randint(1, 20), rng.randint(2, 9)
  return (
      f"Print the result of ({a} plus {b}) times {c}.",
      f"print(({a} + {b}) * {c})",
  )


def _f2_var(rng):
  a, b = rng.randint(2, 20), rng.randint(2, 20)
  return (
      f"Set x to {a} and y to {b}, then print x times y plus x.",
      f"x = {a}\ny = {b}\nprint(x * y + x)",
  )


def _f2_sum3(rng):
  a, b, c = (rng.randint(1, 50) for _ in range(3))
  return (
      f"Set a to {a}, b to {b}, and c to {c}, then print their sum.",
      f"a = {a}\nb = {b}\nc = {c}\nprint(a + b + c)",
  )


def _f2_evenodd(rng):
  n = rng.randint(1, 99)
  return (
      f"Print the word even if {n} is even, otherwise print odd.",
      f"n = {n}\nif n % 2 == 0:\n  print('even')\nelse:\n  print('odd')",
  )


def _f2_maxtwo(rng):
  a, b = rng.randint(1, 99), rng.randint(1, 99)
  while b == a:
    b = rng.randint(1, 99)
  return (
      f"Set a to {a} and b to {b}, then print the larger of the two.",
      f"a = {a}\nb = {b}\nif a > b:\n  print(a)\nelse:\n  print(b)",
  )


def _f2_absdiff(rng):
  a, b = rng.randint(1, 99), rng.randint(1, 99)
  return (
      f"Set a to {a} and b to {b}, then print the absolute value of a minus b.",
      f"a = {a}\nb = {b}\nprint(abs(a - b))",
  )


def _f3_sum_to_n(rng):
  n = rng.randint(5, 120)
  return (
      f"Print the sum of the integers from 1 to {n} (inclusive).",
      f"total = 0\nfor i in range(1, {n + 1}):\n  total += i\nprint(total)",
  )


def _f3_count_lines(rng):
  n = rng.randint(3, 8)
  return (
      f"Print the numbers 1 through {n} (inclusive), one per line.",
      f"for i in range(1, {n + 1}):\n  print(i)",
  )


def _f3_countdown(rng):
  n = rng.randint(3, 9)
  return (
      f"Using a while loop, print the numbers {n}, {n - 1}, ..., 1 each on its "
      "own line (counting down).",
      f"n = {n}\nwhile n >= 1:\n  print(n)\n  n -= 1",
  )


def _f3_factorial_loop(rng):
  n = rng.randint(3, 8)
  return (
      f"Print the product of the integers from 1 to {n}.",
      f"p = 1\nfor i in range(1, {n + 1}):\n  p *= i\nprint(p)",
  )


def _f3_sum_evens(rng):
  n = rng.randint(5, 20) * 2
  return (
      f"Print the sum of all even integers from 1 to {n} (inclusive).",
      f"total = 0\nfor i in range(1, {n + 1}):\n  if i % 2 == 0:\n"
      f"    total += i\nprint(total)",
  )


def _f3_reverse(rng):
  w = rng.choice(_WORDS)
  return f"Print the string {w} reversed.", f"print('{w}'[::-1])"


def _f3_count_vowels(rng):
  w = rng.choice(_WORDS)
  return (
      f"Count how many vowels (a, e, i, o, u) are in the string {w} "
      "and print that count.",
      f"word = '{w}'\ncount = 0\nfor ch in word:\n  if ch in 'aeiou':\n"
      "    count += 1\nprint(count)",
  )


def _f3_fizzbuzz(rng):
  n = rng.randint(10, 20)
  return (
      f"For each integer i from 1 to {n} (inclusive), print one line: Fizz if i "
      "is divisible by 3 and not 5, Buzz if divisible by 5 and not 3, FizzBuzz "
      "if divisible by both, otherwise the number i itself.",
      f"for i in range(1, {n + 1}):\n"
      "  if i % 3 == 0 and i % 5 == 0:\n    print('FizzBuzz')\n"
      "  elif i % 3 == 0:\n    print('Fizz')\n"
      "  elif i % 5 == 0:\n    print('Buzz')\n"
      "  else:\n    print(i)",
  )


def _f4_fib(rng):
  n = rng.randint(5, 15)
  return (
      f"Define a function fib with fib(0)=0, fib(1)=1, and fib(n)=fib(n-1)+"
      f"fib(n-2) for n>=2. Print fib({n}).",
      "def fib(n):\n  if n < 2:\n    return n\n"
      f"  return fib(n - 1) + fib(n - 2)\nprint(fib({n}))",
  )


def _f4_factorial(rng):
  n = rng.randint(3, 9)
  return (
      f"Define a recursive factorial function (with factorial(0)=1) and print "
      f"factorial({n}).",
      "def factorial(n):\n  if n == 0:\n    return 1\n"
      f"  return n * factorial(n - 1)\nprint(factorial({n}))",
  )


_IS_PRIME_DEF = (
    "def is_prime(n):\n  if n < 2:\n    return False\n  d = 2\n"
    "  while d * d <= n:\n    if n % d == 0:\n      return False\n    d += 1\n"
    "  return True\n"
)


def _f4_isprime(rng):
  n = rng.randint(2, 60)
  return (
      f"Define a function is_prime(n) that returns True if n is prime and False "
      f"otherwise, then print is_prime({n}).",
      f"{_IS_PRIME_DEF}print(is_prime({n}))",
  )


def _f4_gcd(rng):
  a, b = rng.randint(6, 99), rng.randint(6, 99)
  return (
      f"Define a function gcd(a, b) using the Euclidean algorithm and print "
      f"gcd({a}, {b}).",
      "def gcd(a, b):\n  while b != 0:\n    a, b = b, a % b\n  return a\n"
      f"print(gcd({a}, {b}))",
  )


def _f4_sumdigits(rng):
  n = rng.randint(100, 99999)
  return (
      f"Define a function sum_digits(n) that returns the sum of the decimal "
      f"digits of a non-negative integer n, then print sum_digits({n}).",
      "def sum_digits(n):\n  total = 0\n  while n > 0:\n    total += n % 10\n"
      f"    n //= 10\n  return total\nprint(sum_digits({n}))",
  )


def _f4_power(rng):
  a, b = rng.randint(2, 6), rng.randint(2, 6)
  return (
      f"Define a recursive function power(base, exp) that computes base raised to "
      f"exp for a non-negative integer exp (with power(base, 0)=1), then print "
      f"power({a}, {b}).",
      "def power(base, exp):\n  if exp == 0:\n    return 1\n"
      f"  return base * power(base, exp - 1)\nprint(power({a}, {b}))",
  )


def _f4_triangular(rng):
  n = rng.randint(5, 50)
  return (
      f"Define a function triangular(n) that returns the nth triangular number "
      f"(the sum 1+2+...+n), then print triangular({n}).",
      f"def triangular(n):\n  return n * (n + 1) // 2\nprint(triangular({n}))",
  )


def _f4_collatz(rng):
  n = rng.randint(2, 27)
  return (
      f"Define a function collatz_steps(n) that counts how many steps it takes to "
      "reach 1 from n, where each step replaces n with n//2 if n is even or 3*n+1 "
      f"if n is odd. Print collatz_steps({n}).",
      "def collatz_steps(n):\n  steps = 0\n  while n != 1:\n"
      "    if n % 2 == 0:\n      n //= 2\n    else:\n      n = 3 * n + 1\n"
      f"    steps += 1\n  return steps\nprint(collatz_steps({n}))",
  )


def _f0_float(rng):
  a = round(rng.uniform(0.1, 99.99), 2)
  return f"Print the number {a}.", f"print({a})"


def _f0_bool(rng):
  b = rng.choice((True, False))
  return f"Print the boolean value {b}.", f"print({b})"


def _f0_upper_letter(rng):
  c = rng.choice(_UPPERCASE)
  return f"Print the single letter {c} (uppercase, no quotes).", f"print('{c}')"


def _f0_phrase(rng):
  phrase = rng.choice(_PHRASES)
  return f"Print exactly: {phrase}", f"print('{phrase}')"


def _f1_truediv(rng):
  b = rng.randint(2, 9)
  a = rng.randint(1, 99)
  return (
      f"Print the result of {a} divided by {b} using true division (a decimal).",
      f"print({a} / {b})",
  )


def _f1_precedence(rng):
  a, b, c = rng.randint(1, 20), rng.randint(2, 12), rng.randint(2, 12)
  return (
      f"Print the result of {a} plus {b} times {c} (using normal operator "
      "precedence).",
      f"print({a} + {b} * {c})",
  )


def _f2_ternary(rng):
  n = rng.randint(-20, 20)
  return (
      f"Set n to {n}. Print the word positive if n is greater than 0, otherwise "
      "print nonpositive.",
      f"n = {n}\nprint('positive' if n > 0 else 'nonpositive')",
  )


def _f2_swap(rng):
  a, b = rng.randint(1, 50), rng.randint(1, 50)
  return (
      f"Set a to {a} and b to {b}, swap them using tuple assignment, then print "
      "a and b on one line separated by a single space.",
      f"a = {a}\nb = {b}\na, b = b, a\nprint(a, b)",
  )


_FIZZBUZZ_BODY = (
    "if n % 3 == 0 and n % 5 == 0:\n  print('FizzBuzz')\n"
    "elif n % 3 == 0:\n  print('Fizz')\n"
    "elif n % 5 == 0:\n  print('Buzz')\n"
    "else:\n  print(n)"
)


def _f2_fizzbuzz_single(rng):
  n = rng.randint(1, 30)
  return (
      f"Set n to {n}. If n is divisible by both 3 and 5 print FizzBuzz, else if "
      "divisible by 3 print Fizz, else if divisible by 5 print Buzz, else print "
      "n.",
      f"n = {n}\n{_FIZZBUZZ_BODY}",
  )


def _f2_grade(rng):
  score = rng.randint(0, 100)
  return (
      f"Set score to {score}. Print A if score is at least 90, B if at least 80, "
      "C if at least 70, otherwise F.",
      f"score = {score}\nif score >= 90:\n  print('A')\n"
      "elif score >= 80:\n  print('B')\nelif score >= 70:\n  print('C')\n"
      "else:\n  print('F')",
  )


def _f3_join_range(rng):
  n = rng.randint(3, 8)
  return (
      f"Print the numbers 1 through {n} (inclusive) on a single line separated by "
      "commas, with no spaces.",
      f"print(','.join([str(i) for i in range(1, {n + 1})]))",
  )


def _f3_max_in_list(rng):
  nums = [rng.randint(1, 30) for _ in range(5)]
  return (
      f"Given the list {nums}, print its largest element using a loop (do not use "
      "the built-in max).",
      f"nums = {nums}\nbest = nums[0]\nfor x in nums:\n  if x > best:\n"
      "    best = x\nprint(best)",
  )


def _f4_count_primes(rng):
  n = rng.randint(10, 40)
  return (
      "Define a function is_prime(n) and use it to count how many integers from "
      f"2 to {n} (inclusive) are prime, then print that count.",
      f"{_IS_PRIME_DEF}count = 0\nfor k in range(2, {n + 1}):\n"
      "  if is_prime(k):\n    count += 1\nprint(count)",
  )


# --- Tier 5: hard / compositional families -------------------------------------
#
# Multi-line, edge-case-heavy programs that compose several skills. The fixed
# tier-5 eval tasks (coding_tasks.py) are saturated by SFT only if the model
# nails every separator / edge case on a single greedy attempt; these randomized
# families give RL the same compositional surface to train on (and a varying gold
# so a constant can't win a group).


def _f5_bubble_sort(rng):
  k = rng.randint(4, 7)
  nums = [rng.randint(1, 30) for _ in range(k)]
  return (
      f"Sort the list {nums} into ascending order using bubble sort (repeatedly "
      "swap adjacent out-of-order pairs; do not use the built-in sorted), then "
      "print the resulting list.",
      f"nums = {nums}\nn = len(nums)\nfor i in range(n):\n"
      "  for j in range(n - 1 - i):\n    if nums[j] > nums[j + 1]:\n"
      "      nums[j], nums[j + 1] = nums[j + 1], nums[j]\nprint(nums)",
  )


def _f5_second_largest(rng):
  # Distinct values so "second largest" is unambiguous.
  nums = rng.sample(range(1, 60), rng.randint(4, 7))
  return (
      f"Given the list {nums}, print its second largest value.",
      f"nums = {nums}\ns = sorted(nums)\nprint(s[-2])",
  )


def _f5_digital_root(rng):
  n = rng.randint(100, 999999)
  return (
      f"Compute the digital root of {n}: repeatedly replace the number with the "
      "sum of its decimal digits until a single digit remains, then print that "
      "digit.",
      f"n = {n}\nwhile n >= 10:\n  t = 0\n  while n > 0:\n    t += n % 10\n"
      "    n //= 10\n  n = t\nprint(n)",
  )


def _f5_nth_prime(rng):
  k = rng.randint(3, 20)
  return (
      f"Print the {k}th prime number (the 1st prime is 2, the 2nd is 3, and so "
      "on).",
      f"target = {k}\ncount = 0\nn = 1\nwhile count < target:\n  n += 1\n"
      "  is_p = True\n  d = 2\n  while d * d <= n:\n    if n % d == 0:\n"
      "      is_p = False\n      break\n    d += 1\n  if is_p:\n    count += 1\n"
      "print(n)",
  )


def _f5_dec_to_binary(rng):
  n = rng.randint(1, 255)
  return (
      f"Print the binary representation of {n} as a string of 0s and 1s, with no "
      "leading zeros and no prefix (compute it manually with repeated division "
      "by 2; do not use bin).",
      f"n = {n}\nbits = ''\nif n == 0:\n  bits = '0'\nwhile n > 0:\n"
      "  bits = str(n % 2) + bits\n  n //= 2\nprint(bits)",
  )


def _f5_reverse_words(rng):
  k = rng.randint(3, 6)
  s = " ".join(rng.choice(_PHRASE_WORDS) for _ in range(k))
  return (
      f"Reverse the order of the words in the sentence '{s}' and print the "
      "result as a single space-separated line (the words themselves are not "
      "reversed, only their order).",
      f"s = '{s}'\nparts = s.split(' ')\nprint(' '.join(parts[::-1]))",
  )


def _f5_most_common_word(rng):
  # A unique mode: one word repeated 3x, three others once each (in fixed slots
  # so the winner is also first), guaranteeing an unambiguous gold.
  pool = rng.sample(_PHRASE_WORDS, 4)
  winner = pool[0]
  words = [winner, pool[1], winner, pool[2], winner, pool[3]]
  s = " ".join(words)
  return (
      f"In the sentence '{s}', find and print the word that appears the most "
      "times (on a tie, print the one that appears first in the sentence).",
      f"s = '{s}'\nwords = s.split(' ')\nbest = words[0]\nbest_count = 0\n"
      "for w in words:\n  c = 0\n  for x in words:\n    if x == w:\n      c += 1\n"
      "  if c > best_count:\n    best_count = c\n    best = w\nprint(best)",
  )


def _f5_run_length_encode(rng):
  chars = rng.sample(_LOWERCASE, rng.randint(2, 4))
  s = "".join(c * rng.randint(1, 4) for c in chars)
  return (
      f"Run-length encode the string '{s}' by replacing each run of a repeated "
      "character with that character followed by the run length, and print the "
      "result (for example 'aaabb' becomes 'a3b2').",
      f"s = '{s}'\nout = ''\ni = 0\nn = len(s)\nwhile i < n:\n  c = s[i]\n"
      "  k = 0\n  while i < n and s[i] == c:\n    k += 1\n    i += 1\n"
      "  out += c + str(k)\nprint(out)",
  )


def _f5_caesar_shift(rng):
  L = rng.randint(3, 6)
  s = "".join(rng.choice(_LOWERCASE) for _ in range(L))
  k = rng.randint(1, 25)
  return (
      f"Apply a Caesar cipher to the lowercase string '{s}', shifting each "
      f"letter forward by {k} positions in the alphabet and wrapping around from "
      "z back to a, then print the result.",
      f"s = '{s}'\nk = {k}\nalpha = 'abcdefghijklmnopqrstuvwxyz'\nout = ''\n"
      "for ch in s:\n  idx = alpha.find(ch)\n  out += alpha[(idx + k) % 26]\n"
      "print(out)",
  )


def _f5_right_triangle(rng):
  n = rng.randint(2, 7)
  return (
      f"Print a right triangle of asterisks with {n} rows: the first row has 1 "
      f"asterisk, the second has 2, and so on up to {n} asterisks on the last "
      "row, each row on its own line.",
      f"n = {n}\nfor i in range(1, n + 1):\n  print('*' * i)",
  )


def _f5_mult_table_row(rng):
  k = rng.randint(2, 12)
  return (
      f"Print the multiplication table row for {k}: the values {k}*1, {k}*2, "
      f"..., {k}*10 on a single line separated by single spaces.",
      f"k = {k}\nparts = []\nfor i in range(1, 11):\n  parts.append(str(k * i))\n"
      "print(' '.join(parts))",
  )


def _f5_sum_of_squares(rng):
  nums = [rng.randint(1, 12) for _ in range(rng.randint(3, 6))]
  return (
      f"Print the sum of the squares of the numbers in the list {nums} (that is, "
      "each number multiplied by itself, all added together).",
      f"nums = {nums}\ntotal = 0\nfor x in nums:\n  total += x * x\nprint(total)",
  )


def _f5_gcd_of_list(rng):
  base = rng.randint(2, 9)
  nums = [base * rng.randint(2, 9) for _ in range(rng.randint(2, 4))]
  return (
      "Define a function gcd(a, b) using the Euclidean algorithm, then use it to "
      f"compute and print the greatest common divisor of every number in the "
      f"list {nums}.",
      "def gcd(a, b):\n  while b != 0:\n    a, b = b, a % b\n  return a\n"
      f"nums = {nums}\ng = nums[0]\nfor x in nums:\n  g = gcd(g, x)\nprint(g)",
  )


def _f5_fib_list(rng):
  n = rng.randint(5, 12)
  return (
      f"Print the first {n} Fibonacci numbers (starting from 0, 1) on a single "
      "line separated by commas with no spaces.",
      f"n = {n}\na, b = 0, 1\nparts = []\nfor i in range(n):\n"
      "  parts.append(str(a))\n  a, b = b, a + b\nprint(','.join(parts))",
  )


def _f5_collatz_sequence(rng):
  n = rng.randint(3, 27)
  return (
      f"Print the full Collatz sequence starting from {n} down to 1 on a single "
      "line separated by commas with no spaces, where each step replaces n with "
      f"n//2 if n is even or 3*n+1 if n is odd (include both the starting {n} and "
      "the final 1).",
      f"n = {n}\nparts = []\nwhile n != 1:\n  parts.append(str(n))\n"
      "  if n % 2 == 0:\n    n //= 2\n  else:\n    n = 3 * n + 1\n"
      "parts.append('1')\nprint(','.join(parts))",
  )


def _f5_integer_average(rng):
  nums = [rng.randint(1, 99) for _ in range(rng.randint(3, 7))]
  return (
      f"Print the integer (floor) average of the numbers in the list {nums} "
      "(their sum divided by their count using floor division).",
      f"nums = {nums}\nprint(sum(nums) // len(nums))",
  )


def _f5_reverse_integer(rng):
  # Leading digit nonzero so the reversed integer has no ambiguous trailing zero.
  digits = [str(rng.randint(1, 9))] + [
      str(rng.randint(0, 9)) for _ in range(rng.randint(2, 4))
  ]
  n = int("".join(digits))
  return (
      f"Reverse the digits of the integer {n} and print the resulting integer.",
      f"n = {n}\nrev = 0\nwhile n > 0:\n  rev = rev * 10 + n % 10\n  n //= 10\n"
      "print(rev)",
  )


def _f5_is_palindrome(rng):
  w = rng.choice(_PALINDROME_WORDS)
  return (
      f"Print yes if the string '{w}' reads the same forwards and backwards, "
      "otherwise print no.",
      f"s = '{w}'\nif s == s[::-1]:\n  print('yes')\nelse:\n  print('no')",
  )


FAMILIES: Tuple[Family, ...] = (
    Family("f0_int", 0, _f0_int),
    Family("f0_word", 0, _f0_word),
    Family("f0_float", 0, _f0_float),
    Family("f0_bool", 0, _f0_bool),
    Family("f0_upper_letter", 0, _f0_upper_letter),
    Family("f0_phrase", 0, _f0_phrase),
    Family("f1_binop", 1, _f1_binop),
    Family("f1_floordiv", 1, _f1_floordiv),
    Family("f1_mod", 1, _f1_mod),
    Family("f1_power", 1, _f1_power),
    Family("f1_parens", 1, _f1_parens),
    Family("f1_truediv", 1, _f1_truediv),
    Family("f1_precedence", 1, _f1_precedence),
    Family("f2_var", 2, _f2_var),
    Family("f2_sum3", 2, _f2_sum3),
    Family("f2_evenodd", 2, _f2_evenodd),
    Family("f2_maxtwo", 2, _f2_maxtwo),
    Family("f2_absdiff", 2, _f2_absdiff),
    Family("f2_ternary", 2, _f2_ternary),
    Family("f2_swap", 2, _f2_swap),
    Family("f2_fizzbuzz_single", 2, _f2_fizzbuzz_single),
    Family("f2_grade", 2, _f2_grade),
    Family("f3_sum_to_n", 3, _f3_sum_to_n),
    Family("f3_count_lines", 3, _f3_count_lines),
    Family("f3_countdown", 3, _f3_countdown),
    Family("f3_factorial_loop", 3, _f3_factorial_loop),
    Family("f3_sum_evens", 3, _f3_sum_evens),
    Family("f3_reverse", 3, _f3_reverse),
    Family("f3_count_vowels", 3, _f3_count_vowels),
    Family("f3_fizzbuzz", 3, _f3_fizzbuzz),
    Family("f3_join_range", 3, _f3_join_range),
    Family("f3_max_in_list", 3, _f3_max_in_list),
    Family("f4_fib", 4, _f4_fib),
    Family("f4_factorial", 4, _f4_factorial),
    Family("f4_isprime", 4, _f4_isprime),
    Family("f4_gcd", 4, _f4_gcd),
    Family("f4_sumdigits", 4, _f4_sumdigits),
    Family("f4_power", 4, _f4_power),
    Family("f4_triangular", 4, _f4_triangular),
    Family("f4_collatz", 4, _f4_collatz),
    Family("f4_count_primes", 4, _f4_count_primes),
    Family("f5_bubble_sort", 5, _f5_bubble_sort),
    Family("f5_second_largest", 5, _f5_second_largest),
    Family("f5_digital_root", 5, _f5_digital_root),
    Family("f5_nth_prime", 5, _f5_nth_prime),
    Family("f5_dec_to_binary", 5, _f5_dec_to_binary),
    Family("f5_reverse_words", 5, _f5_reverse_words),
    Family("f5_most_common_word", 5, _f5_most_common_word),
    Family("f5_run_length_encode", 5, _f5_run_length_encode),
    Family("f5_caesar_shift", 5, _f5_caesar_shift),
    Family("f5_right_triangle", 5, _f5_right_triangle),
    Family("f5_mult_table_row", 5, _f5_mult_table_row),
    Family("f5_sum_of_squares", 5, _f5_sum_of_squares),
    Family("f5_gcd_of_list", 5, _f5_gcd_of_list),
    Family("f5_fib_list", 5, _f5_fib_list),
    Family("f5_collatz_sequence", 5, _f5_collatz_sequence),
    Family("f5_integer_average", 5, _f5_integer_average),
    Family("f5_reverse_integer", 5, _f5_reverse_integer),
    Family("f5_is_palindrome", 5, _f5_is_palindrome),
)


def families_for_tiers(tiers: Tuple[int, ...]) -> List[Family]:
  """The families whose tier is in ``tiers`` (the curriculum selector)."""
  return [f for f in FAMILIES if f.tier in tiers]


def sample_task(
    rng: random.Random, tiers: Tuple[int, ...]
) -> Tuple[str, str, str]:
  """Samples one ``(prompt, solution, gold)`` from the enabled-tier families.

  The gold is computed by actually running the solution through the interpreter,
  so it is always consistent with what the grader will compare against.
  """
  fam = rng.choice(families_for_tiers(tiers))
  prompt, solution = fam.sample(rng)
  gold = micropython.run(solution, max_steps=_EVAL_MAX_STEPS).stdout
  return prompt, solution, gold


# --- SFT (execution-format) transcripts ----------------------------------------


def code_segments(rng: random.Random, tiers: Tuple[int, ...]):
  """One SFT transcript: ``Task: <prompt>`` (masked) + ``<solution> END`` (train).

  Consumed by :func:`agentic_sft.run_sft_warmup` with ``prompt_prefix=CODE_FEWSHOT``
  (prepended masked), so the model is trained to emit the program + the ``END``
  sentinel given the few-shot format -- the format/mapping warm-up before RL.
  """
  prompt, solution, _ = sample_task(rng, tiers)
  return [
      (f"Task: {prompt}\n", 0),
      (f"{solution}\nEND\n", 1),
  ]


# --- The grain dataset ---------------------------------------------------------


class _CodeSource(grain.RandomAccessDataSource):
  """A grain source of ``(prompt_text, gold)`` coding rows (pre-generated)."""

  def __init__(self, n: int, seed: int, tiers: Tuple[int, ...]):
    rng = random.Random(seed)
    self._rows: List[Tuple[str, str]] = []
    for _ in range(n):
      prompt, _solution, gold = sample_task(rng, tiers)
      self._rows.append((build_code_prompt(prompt), gold))

  def __len__(self) -> int:
    return len(self._rows)

  def __getitem__(self, idx: int) -> Tuple[str, str]:
    return self._rows[idx]


def build_code_dataset(
    n: int, seed: int, batch_size: int, tiers: Tuple[int, ...]
) -> grain.MapDataset:
  """Builds a batched grain dataset with ``prompts`` + ``answer`` columns.

  Same M2-proven grain shape as :func:`arithmetic.build_arithmetic_dataset`
  (tuple source -> ``.batch`` collates field-wise into single numpy-array leaves);
  the learner forwards the ``answer`` (gold stdout) column to the reward/metric.
  """
  source = _CodeSource(n, seed, tiers)

  def _to_columns(batch):
    prompts, answers = batch
    return {"prompts": prompts, "answer": answers}

  return grain.MapDataset.source(source).batch(batch_size).map(_to_columns)


# --- Eval on the fixed 50 tasks ------------------------------------------------


def strip_answer_hint(prompt: str, answer: str) -> str:
  """Removes trailing parenthetical clauses that LEAK the gold output.

  The fixed eval prompts append hints like ``(The answer is 55.)`` or ``(13 is
  prime, so print True.)`` to be unambiguous; left in, a model could hardcode the
  leaked value instead of computing. We strip a trailing ``(...)`` clause iff its
  interior contains a gold output line -- so output-leaking hints go, while
  input-defining / clarifying parens (``(inclusive)``, ``(n is 15.)``,
  ``fib(10)``) stay. Keeps the eval prompt style close to the (hint-free) training
  prompts (invariant D).
  """
  golds = [g for g in answer.strip().split("\n") if g]
  out = prompt
  while True:
    match = re.search(r"\s*\(([^()]*)\)\.?\s*$", out)
    if not match:
      break
    interior = match.group(1)
    if any(g and g in interior for g in golds):
      out = out[: match.start()].rstrip()
    else:
      break
  return out


@dataclasses.dataclass
class TaskEvalRow:
  """The eval outcome for one fixed task."""

  task_id: str
  tier: int
  solved: bool
  ran_ok: bool
  program: str
  output: str


@dataclasses.dataclass
class CodingEvalResult:
  """Aggregate eval over the fixed 50 tasks."""

  rows: List[TaskEvalRow]

  @property
  def solved(self) -> int:
    return sum(1 for r in self.rows if r.solved)

  @property
  def total(self) -> int:
    return len(self.rows)

  def per_tier(self) -> Dict[int, Tuple[int, int]]:
    """tier -> (solved, total)."""
    out: Dict[int, List[int]] = {}
    for r in self.rows:
      acc = out.setdefault(r.tier, [0, 0])
      acc[0] += int(r.solved)
      acc[1] += 1
    return {t: (s, n) for t, (s, n) in out.items()}

  def summary(self) -> str:
    lines = [f"solve {self.solved}/{self.total}"]
    for tier in sorted(self.per_tier()):
      s, n = self.per_tier()[tier]
      lines.append(f"  tier {tier}: {s}/{n}")
    return "\n".join(lines)


def evaluate_completions(tasks, completions) -> CodingEvalResult:
  """Grades pre-generated completions for the fixed tasks (dep-free)."""
  rows: List[TaskEvalRow] = []
  for task, completion in zip(tasks, completions):
    program = extract_program(str(completion))
    result = micropython.run(program, max_steps=_EVAL_MAX_STEPS)
    solved = result.ok and result.stdout == task.answer
    rows.append(
        TaskEvalRow(
            task_id=task.id,
            tier=task.tier,
            solved=bool(solved),
            ran_ok=bool(result.ok),
            program=program,
            output=result.stdout,
        )
    )
  return CodingEvalResult(rows=rows)


def eval_prompts(tasks) -> List[str]:
  """The eval prompt strings for the fixed tasks (hints stripped)."""
  return [
      build_code_prompt(strip_answer_hint(t.prompt, t.answer)) for t in tasks
  ]


def greedy_completions(
    model,
    tokenizer,
    prompts: List[str],
    *,
    max_new_tokens: int,
    max_prompt_length: int,
    mesh=None,
    chunk_size: int = 25,
) -> List[str]:
  """Greedy-decodes completions for ``prompts`` via tunix's vanilla Sampler.

  Modeled on ``_validate_m4_arithmetic.py``'s sampler use: ``temperature=0.0`` is
  greedy (argmax). The actor is sharded on ``mesh``, so generation runs inside the
  mesh context. Prompts are chunked to bound the cache. ``eos_tokens=[EOS]`` will
  not fire (the base LM never emits EOS); generation runs to ``max_new_tokens``
  and :func:`extract_program` cuts at the ``END`` sentinel.
  """
  from tunix.generate import sampler as sampler_lib  # lazy: tunix only at eval

  from models.delphi_qwen3 import DELPHI_EOS_ID

  cache_config = sampler_lib.CacheConfig(
      cache_size=max_prompt_length + max_new_tokens + 8,
      num_layers=model.config.num_layers,
      num_kv_heads=model.config.num_kv_heads,
      head_dim=model.config.head_dim,
  )
  sampler = sampler_lib.Sampler(
      transformer=model, tokenizer=tokenizer, cache_config=cache_config
  )

  import contextlib

  ctx = mesh if mesh is not None else contextlib.nullcontext()
  out: List[str] = []
  with ctx:
    for i in range(0, len(prompts), chunk_size):
      batch = prompts[i : i + chunk_size]
      result = sampler(
          input_strings=batch,
          max_generation_steps=max_new_tokens,
          max_prompt_length=max_prompt_length,
          echo=False,
          eos_tokens=[DELPHI_EOS_ID],
          temperature=0.0,
          seed=0,
      )
      out.extend(result.text)
  return out


def evaluate_tasks(
    model,
    tokenizer,
    tasks,
    *,
    max_new_tokens: int = 160,
    max_prompt_length: int = 384,
    mesh=None,
) -> CodingEvalResult:
  """Greedy-evaluates ``model`` on the fixed ``tasks`` and grades each program."""
  completions = greedy_completions(
      model,
      tokenizer,
      eval_prompts(tasks),
      max_new_tokens=max_new_tokens,
      max_prompt_length=max_prompt_length,
      mesh=mesh,
  )
  return evaluate_completions(tasks, completions)

