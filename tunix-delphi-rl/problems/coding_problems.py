# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test-case-graded coding problems for the RL curriculum (issue #8).

This is the *new* problem representation that replaces the single-gold
"write a script whose stdout equals one fixed string" setup (:mod:`coding_env` /
:mod:`coding_tasks`). Here the agent is asked to write a function
``def solve(...)`` and that function is graded against **N independent test
cases**, the canonical RLVR-for-code regime. The motivation (see
``CURRICULUM_DESIGN.md`` §1, §2) is to give Dr.GRPO a *dense, continuous* reward
signal: a group of sampled programs that each pass a different fraction of the
tests has non-zero intra-group reward variance even when no sample is perfect,
which is exactly the gradient the old binary reward starved.

The pieces:

  * :class:`Problem` -- one graded instance: an ``id``, a ``level`` (1..9), a
    ``family`` key, a human ``prompt``, a small set of ``public_tests`` shown to
    the agent, and a larger set of graded ``hidden_tests``.
  * The **leveled families** -- procedurally parameterised ``solve`` tasks of
    increasing composition depth (3-4 per level, 9 levels). Levels 7-9 are the
    multi-stage (compositional) families: each ``solve`` computes an intermediate
    result (primes <= n, a deduped/sorted set, per-word counts, a factorial, a
    parsed expression, a Collatz trajectory) and then reduces or transforms it,
    so an unreliable first attempt tends to be partially right (a non-trivial
    pass@k > pass@1 gap for RL to sharpen). Each family ships a
    ``reference`` (a correct ``def solve(...)`` program *valid under
    micropython*) and a ``gen(rng)`` instance generator that produces varied
    inputs including edge cases (empty list, ``n=0/1``, negatives, duplicates,
    ties) so a memorised constant fails the hidden tests.
  * :func:`grade_problem` / :func:`problem_reward` -- run the submitted program
    against every (public + hidden) test through :mod:`micropython` and return
    the dense components ``has_code`` / ``ran_ok`` / ``frac_passed`` / ``exact``
    and the weighted reward ``0.10*has_code + 0.20*ran_ok + 0.70*frac_passed``.

Crucial correctness detail (grading mechanics): the expected output of a test is
computed by running the family's ``reference`` *through micropython itself* (NOT
host Python), so the print-formatting of the gold matches the engine the agent's
program runs in. micropython does **not** support ``*args`` calls
(``solve(*(5,))`` raises ``UnsupportedSyntax``), so a test ``(args, expected)``
is executed by expanding the argument tuple into a literal call --
``solve(3, 4)`` / ``solve([1, 2, 3])`` -- via ``", ".join(repr(a) for a in
args)``, which round-trips cleanly through the interpreter's parser.

Dropped from the design table relative to ``CURRICULUM_DESIGN.md``:

  * **``most_common_word``** keeps the design's name but is implemented with the
    nested-loop counting trick (no ``dict``), because micropython does not
    support dict literals (``UnsupportedSyntax: Dict``). Tie-break: the word that
    appears first in the sentence wins, matching ``coding_tasks`` t5_07.
  * **``caesar``** avoids ``ord``/``chr`` (not whitelisted) by indexing into an
    explicit alphabet string with ``.find`` and ``% 26``.

No family was dropped outright; every level keeps 3-4 families.

Run ``JAX_PLATFORMS=cpu uv run pytest tests/test_coding_problems.py`` to validate
that every reference passes all its own tests, that the tests discriminate against
wrong and empty programs, and that partial credit produces a strictly-interior reward.
"""

from __future__ import annotations

import dataclasses
import random
from typing import Callable

import environments.micropython as micropython

NUM_LEVELS = 9

# Step budget for grading a single test execution. The heaviest references are
# the L6 nth-prime / collatz searches, which take ~2000 AST steps on the largest
# generated instances (measured); 4000 gives comfortable headroom for any correct
# submission while still bounding a runaway / infinite-loop program (which hits
# the limit and counts as a non-ok run rather than hanging). This is the
# ``max_steps`` default for grade_problem / format_test_feedback -- the design
# table's nominal 600 is too small for the L6 search families.
_GRADE_MAX_STEPS = 4000

# How much feedback to show the agent per multi-turn round (chars). Matches the
# convention in :mod:`coding_agent_env`.
_MAX_FEEDBACK_CHARS = 400

# Public/hidden test-count contract (the API minimums). We generate >=7 tests per
# instance: the first _N_PUBLIC become public, the rest hidden (>= _MIN_HIDDEN).
_N_PUBLIC = 2
_MIN_HIDDEN = 5
_MIN_TESTS = _N_PUBLIC + _MIN_HIDDEN


@dataclasses.dataclass
class Problem:
  """One test-case-graded coding problem instance.

  Attributes:
    id: Stable slug, unique per (family, instance seed), e.g.
      ``"l5_second_largest#a3f1"``.
    level: Curriculum level in ``1..NUM_LEVELS``.
    family: Family key, e.g. ``"l5_second_largest"``.
    prompt: The human task text (signature + behaviour + public examples + the
      END-line instruction), produced by :func:`format_problem_prompt`.
    public_tests: ``(args_tuple, expected_stdout)`` pairs shown to the agent
      (``<= _N_PUBLIC``).
    hidden_tests: ``(args_tuple, expected_stdout)`` pairs that are graded but
      never revealed (``>= _MIN_HIDDEN``).
  """

  id: str
  level: int
  family: str
  prompt: str
  public_tests: list[tuple[tuple, str]]
  hidden_tests: list[tuple[tuple, str]]


# ---------------------------------------------------------------------------
# Family definition: a reference solver + an instance generator + task text.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Family:
  """A parameterised problem family.

  Attributes:
    key: The family key (and ``Problem.family``), e.g. ``"l1_add"``.
    level: The curriculum level the family belongs to.
    signature: The ``solve`` signature as shown in the prompt, e.g.
      ``"solve(a, b)"``.
    description: One-line behaviour description for the prompt.
    reference: A correct ``def solve(...)`` program valid under micropython; used
      to compute every test's expected output (run *through* micropython) and
      asserted in the self-check to pass all its own tests.
    gen: ``gen(rng) -> args_tuple`` producing one varied instance's argument
      tuple. Called once per test, so a single instance draws several distinct
      arg tuples (incl. edge cases) -- the anti-hardcode property.
  """

  key: str
  level: int
  signature: str
  description: str
  reference: str
  gen: Callable[[random.Random], tuple]


# ---------------------------------------------------------------------------
# The leveled families (3-4 per level, 9 levels of increasing depth; levels 7-9
# are the multi-stage / compositional families).
#
# Each ``reference`` was verified to run cleanly under micropython (see the
# module docstring / the CPU self-check below). Generators deliberately include
# edge cases -- n=0/1, empty list, negatives, duplicates, ties -- so the hidden
# tests discriminate a hard-coded constant from a real solution.
# ---------------------------------------------------------------------------

_FAMILIES: list[_Family] = [
    # --- Level 1: return / arithmetic -------------------------------------
    _Family(
        key="l1_add",
        level=1,
        signature="solve(a, b)",
        description="Return the sum of the two integers a and b.",
        reference="def solve(a, b):\n  return a + b",
        gen=lambda rng: (rng.randint(-50, 50), rng.randint(-50, 50)),
    ),
    _Family(
        key="l1_abs_diff",
        level=1,
        signature="solve(a, b)",
        description="Return the absolute value of (a minus b).",
        reference=(
            "def solve(a, b):\n"
            "  d = a - b\n"
            "  if d < 0:\n"
            "    return -d\n"
            "  return d"
        ),
        gen=lambda rng: (rng.randint(-30, 30), rng.randint(-30, 30)),
    ),
    _Family(
        key="l1_negate",
        level=1,
        signature="solve(n)",
        description="Return n with its sign flipped (the negation of n).",
        reference="def solve(n):\n  return -n",
        gen=lambda rng: (rng.randint(-99, 99),),
    ),
    # --- Level 2: one loop / branch ---------------------------------------
    _Family(
        key="l2_sum_1_to_n",
        level=2,
        signature="solve(n)",
        description=(
            "Return the sum of the integers from 1 to n inclusive "
            "(return 0 when n is 0 or negative)."
        ),
        reference=(
            "def solve(n):\n"
            "  total = 0\n"
            "  for i in range(1, n + 1):\n"
            "    total += i\n"
            "  return total"
        ),
        # Include n=0 and n=1 edge cases plus larger values.
        gen=lambda rng: (rng.choice([0, 1, 1, rng.randint(2, 40)]),),
    ),
    _Family(
        key="l2_max_of_list",
        level=2,
        signature="solve(xs)",
        description="Return the largest element of the non-empty list xs.",
        reference=(
            "def solve(xs):\n"
            "  best = xs[0]\n"
            "  for x in xs:\n"
            "    if x > best:\n"
            "      best = x\n"
            "  return best"
        ),
        gen=lambda rng: (_rand_int_list(rng, 1, 8, -20, 20, allow_dups=True),),
    ),
    _Family(
        key="l2_count_evens",
        level=2,
        signature="solve(xs)",
        description="Return how many elements of the list xs are even.",
        reference=(
            "def solve(xs):\n"
            "  c = 0\n"
            "  for x in xs:\n"
            "    if x % 2 == 0:\n"
            "      c += 1\n"
            "  return c"
        ),
        gen=lambda rng: (_rand_int_list(rng, 0, 8, -15, 15, allow_dups=True),),
    ),
    # --- Level 3: basic algorithms ----------------------------------------
    _Family(
        key="l3_factorial",
        level=3,
        signature="solve(n)",
        description="Return n factorial (the product 1*2*...*n; 0! is 1).",
        reference=(
            "def solve(n):\n"
            "  p = 1\n"
            "  for i in range(1, n + 1):\n"
            "    p *= i\n"
            "  return p"
        ),
        gen=lambda rng: (rng.choice([0, 1, 1, rng.randint(2, 9)]),),
    ),
    _Family(
        key="l3_fib",
        level=3,
        signature="solve(n)",
        description=(
            "Return the nth Fibonacci number with fib(0)=0, fib(1)=1, "
            "fib(n)=fib(n-1)+fib(n-2)."
        ),
        reference=(
            "def solve(n):\n"
            "  if n < 2:\n"
            "    return n\n"
            "  a, b = 0, 1\n"
            "  for i in range(2, n + 1):\n"
            "    a, b = b, a + b\n"
            "  return b"
        ),
        gen=lambda rng: (rng.choice([0, 1, rng.randint(2, 20)]),),
    ),
    _Family(
        key="l3_reverse_list",
        level=3,
        signature="solve(xs)",
        description="Return a new list with the elements of xs in reverse order.",
        reference="def solve(xs):\n  return xs[::-1]",
        gen=lambda rng: (_rand_int_list(rng, 0, 7, -20, 20, allow_dups=True),),
    ),
    _Family(
        key="l3_gcd",
        level=3,
        signature="solve(a, b)",
        description=(
            "Return the greatest common divisor of the positive integers a and b "
            "(Euclidean algorithm)."
        ),
        reference=(
            "def solve(a, b):\n"
            "  while b != 0:\n"
            "    a, b = b, a % b\n"
            "  return a"
        ),
        gen=lambda rng: (rng.randint(1, 100), rng.randint(1, 100)),
    ),
    # --- Level 4: simple composition --------------------------------------
    _Family(
        key="l4_is_prime",
        level=4,
        signature="solve(n)",
        description=(
            "Return True if n is a prime number and False otherwise "
            "(n < 2 is not prime)."
        ),
        reference=(
            "def solve(n):\n"
            "  if n < 2:\n"
            "    return False\n"
            "  d = 2\n"
            "  while d * d <= n:\n"
            "    if n % d == 0:\n"
            "      return False\n"
            "    d += 1\n"
            "  return True"
        ),
        # Mix small edge cases (0,1,2) with composites and primes.
        gen=lambda rng: (rng.choice([0, 1, 2, rng.randint(3, 60)]),),
    ),
    _Family(
        key="l4_bubble_sort",
        level=4,
        signature="solve(xs)",
        description=(
            "Return a new list with the elements of xs sorted in ascending "
            "order (you may implement it however you like)."
        ),
        reference=(
            "def solve(xs):\n"
            "  a = xs[:]\n"
            "  n = len(a)\n"
            "  for i in range(n):\n"
            "    for j in range(n - 1 - i):\n"
            "      if a[j] > a[j + 1]:\n"
            "        a[j], a[j + 1] = a[j + 1], a[j]\n"
            "  return a"
        ),
        gen=lambda rng: (_rand_int_list(rng, 0, 7, -20, 20, allow_dups=True),),
    ),
    _Family(
        key="l4_digit_sum",
        level=4,
        signature="solve(n)",
        description=(
            "Return the sum of the decimal digits of the non-negative integer n "
            "(digit_sum(0) is 0)."
        ),
        reference=(
            "def solve(n):\n"
            "  if n == 0:\n"
            "    return 0\n"
            "  total = 0\n"
            "  while n > 0:\n"
            "    total += n % 10\n"
            "    n //= 10\n"
            "  return total"
        ),
        gen=lambda rng: (rng.choice([0, rng.randint(1, 99999)]),),
    ),
    _Family(
        key="l4_count_vowels",
        level=4,
        signature="solve(s)",
        description=(
            "Return how many vowels (a, e, i, o, u) the lowercase string s "
            "contains."
        ),
        reference=(
            "def solve(s):\n"
            "  c = 0\n"
            "  for ch in s:\n"
            "    if ch in 'aeiou':\n"
            "      c += 1\n"
            "  return c"
        ),
        gen=lambda rng: (_rand_word(rng, 0, 9),),
    ),
    # --- Level 5: multi-step ----------------------------------------------
    _Family(
        key="l5_second_largest",
        level=5,
        signature="solve(xs)",
        description=(
            "Return the second-largest value of xs by sorted order: sort xs "
            "ascending and return the second element from the end (so duplicates "
            "of the maximum still count). xs has at least two elements."
        ),
        reference=(
            "def solve(xs):\n"
            "  s = sorted(xs)\n"
            "  return s[-2]"
        ),
        # >=2 elements, with duplicate-of-max ties deliberately seeded.
        gen=lambda rng: (_rand_int_list(rng, 2, 7, -15, 15, allow_dups=True),),
    ),
    _Family(
        key="l5_run_length_encode",
        level=5,
        signature="solve(s)",
        description=(
            "Run-length encode the lowercase string s: replace each maximal run "
            "of a repeated character with that character followed by the run "
            "length (e.g. 'aaabb' -> 'a3b2'). Return the encoded string "
            "('' -> '')."
        ),
        reference=(
            "def solve(s):\n"
            "  out = ''\n"
            "  i = 0\n"
            "  n = len(s)\n"
            "  while i < n:\n"
            "    c = s[i]\n"
            "    k = 0\n"
            "    while i < n and s[i] == c:\n"
            "      k += 1\n"
            "      i += 1\n"
            "    out += c + str(k)\n"
            "  return out"
        ),
        gen=lambda rng: (_rand_runs(rng),),
    ),
    _Family(
        key="l5_digital_root",
        level=5,
        signature="solve(n)",
        description=(
            "Return the digital root of the non-negative integer n: repeatedly "
            "replace n by the sum of its digits until a single digit remains."
        ),
        reference=(
            "def solve(n):\n"
            "  while n >= 10:\n"
            "    t = 0\n"
            "    while n > 0:\n"
            "      t += n % 10\n"
            "      n //= 10\n"
            "    n = t\n"
            "  return n"
        ),
        gen=lambda rng: (rng.choice([0, rng.randint(1, 9), rng.randint(10, 99999)]),),
    ),
    _Family(
        key="l5_caesar",
        level=5,
        signature="solve(s, shift)",
        description=(
            "Apply a Caesar cipher to the lowercase string s: shift each letter "
            "forward by `shift` positions in the alphabet, wrapping from z back "
            "to a. Return the resulting string."
        ),
        reference=(
            "def solve(s, shift):\n"
            "  alpha = 'abcdefghijklmnopqrstuvwxyz'\n"
            "  out = ''\n"
            "  for ch in s:\n"
            "    idx = alpha.find(ch)\n"
            "    out += alpha[(idx + shift) % 26]\n"
            "  return out"
        ),
        gen=lambda rng: (_rand_word(rng, 1, 8), rng.randint(0, 25)),
    ),
    # --- Level 6: harder --------------------------------------------------
    _Family(
        key="l6_nth_prime",
        level=6,
        signature="solve(k)",
        description="Return the kth prime number (the 1st prime is 2, the 2nd is 3).",
        reference=(
            "def solve(k):\n"
            "  count = 0\n"
            "  n = 1\n"
            "  while count < k:\n"
            "    n += 1\n"
            "    is_p = True\n"
            "    d = 2\n"
            "    while d * d <= n:\n"
            "      if n % d == 0:\n"
            "        is_p = False\n"
            "        break\n"
            "      d += 1\n"
            "    if is_p:\n"
            "      count += 1\n"
            "  return n"
        ),
        gen=lambda rng: (rng.randint(1, 15),),
    ),
    _Family(
        key="l6_collatz_len",
        level=6,
        signature="solve(n)",
        description=(
            "Return how many steps it takes to reach 1 from the positive integer "
            "n, where each step replaces n by n//2 if n is even or 3*n+1 if n is "
            "odd (solve(1) is 0)."
        ),
        reference=(
            "def solve(n):\n"
            "  steps = 0\n"
            "  while n != 1:\n"
            "    if n % 2 == 0:\n"
            "      n //= 2\n"
            "    else:\n"
            "      n = 3 * n + 1\n"
            "    steps += 1\n"
            "  return steps"
        ),
        gen=lambda rng: (rng.choice([1, 2, rng.randint(3, 50)]),),
    ),
    _Family(
        key="l6_most_common_word",
        level=6,
        signature="solve(s)",
        description=(
            "Return the word that appears most often in the space-separated "
            "sentence s. On a tie, return the word that appears first in s."
        ),
        # No dict (micropython lacks dict literals); count with a nested loop.
        reference=(
            "def solve(s):\n"
            "  words = s.split(' ')\n"
            "  best = words[0]\n"
            "  best_count = 0\n"
            "  for w in words:\n"
            "    c = 0\n"
            "    for x in words:\n"
            "      if x == w:\n"
            "        c += 1\n"
            "    if c > best_count:\n"
            "      best_count = c\n"
            "      best = w\n"
            "  return best"
        ),
        gen=lambda rng: (_rand_sentence(rng),),
    ),
    _Family(
        key="l6_is_palindrome_sentence",
        level=6,
        signature="solve(s)",
        description=(
            "Return 'yes' if the string s reads the same forwards and backwards "
            "after removing all spaces, otherwise return 'no'."
        ),
        reference=(
            "def solve(s):\n"
            "  t = ''.join(s.split(' '))\n"
            "  if t == t[::-1]:\n"
            "    return 'yes'\n"
            "  return 'no'"
        ),
        gen=lambda rng: (_rand_palindrome_candidate(rng),),
    ),
    # --- Level 7: multi-stage (compositional) -----------------------------
    # Two clearly separate stages: produce an intermediate (primes / distinct
    # set / per-word counts / a factorial) and then reduce or transform it.
    _Family(
        key="l7_sum_primes",
        level=7,
        signature="solve(n)",
        description=(
            "Return the sum of all prime numbers that are <= n "
            "(so solve(10) is 2+3+5+7 = 17; return 0 when n < 2)."
        ),
        # Stage 1: test each m in 2..n for primality. Stage 2: accumulate the sum.
        reference=(
            "def solve(n):\n"
            "  total = 0\n"
            "  m = 2\n"
            "  while m <= n:\n"
            "    is_p = True\n"
            "    d = 2\n"
            "    while d * d <= m:\n"
            "      if m % d == 0:\n"
            "        is_p = False\n"
            "        break\n"
            "      d += 1\n"
            "    if is_p:\n"
            "      total += m\n"
            "    m += 1\n"
            "  return total"
        ),
        # Bias toward n>=5 (and seed 2,3) so a constant 0 fails the hidden tests.
        gen=lambda rng: (
            rng.choice([2, 3, rng.randint(5, 60), rng.randint(5, 60), rng.randint(5, 60)]),
        ),
    ),
    _Family(
        key="l7_second_distinct_sq",
        level=7,
        signature="solve(xs)",
        description=(
            "From the list xs, find the second-largest DISTINCT value (ignore "
            "duplicates), and return its square. xs has at least two distinct "
            "values."
        ),
        # Stage 1: dedupe (preserving membership). Stage 2: sort. Stage 3: pick the
        # second-from-top distinct value. Stage 4: square it.
        reference=(
            "def solve(xs):\n"
            "  uniq = []\n"
            "  for x in xs:\n"
            "    if x not in uniq:\n"
            "      uniq.append(x)\n"
            "  uniq = sorted(uniq)\n"
            "  v = uniq[-2]\n"
            "  return v * v"
        ),
        gen=lambda rng: (_rand_dedup_list(rng, -9, 9),),
    ),
    _Family(
        key="l7_most_vowels_word",
        level=7,
        signature="solve(s)",
        description=(
            "Split the space-separated sentence s into words, count the vowels "
            "(a, e, i, o, u) in each word, and return the word with the most "
            "vowels. On a tie, return the word that appears first."
        ),
        # Stage 1: split into words. Stage 2: per-word vowel count. Stage 3: argmax.
        reference=(
            "def solve(s):\n"
            "  words = s.split(' ')\n"
            "  best = words[0]\n"
            "  best_c = -1\n"
            "  for w in words:\n"
            "    c = 0\n"
            "    for ch in w:\n"
            "      if ch in 'aeiou':\n"
            "        c += 1\n"
            "    if c > best_c:\n"
            "      best_c = c\n"
            "      best = w\n"
            "  return best"
        ),
        gen=lambda rng: (_rand_vowel_sentence(rng),),
    ),
    _Family(
        key="l7_fact_digit_sum",
        level=7,
        signature="solve(n)",
        description=(
            "Return the sum of the decimal digits of n factorial "
            "(compute n! = 1*2*...*n, then add up its digits; 0! is 1)."
        ),
        # Stage 1: compute the factorial. Stage 2: digit-sum that (large) integer.
        reference=(
            "def solve(n):\n"
            "  p = 1\n"
            "  for i in range(1, n + 1):\n"
            "    p *= i\n"
            "  total = 0\n"
            "  while p > 0:\n"
            "    total += p % 10\n"
            "    p //= 10\n"
            "  return total"
        ),
        gen=lambda rng: (rng.choice([0, 1, rng.randint(2, 12)]),),
    ),
    # --- Level 8: deeper composition / light parsing ----------------------
    _Family(
        key="l8_rle_reversed",
        level=8,
        signature="solve(xs)",
        description=(
            "Run-length encode the list xs into a flat list [value, count, "
            "value, count, ...] (one pair per maximal run of equal values), then "
            "return that encoded list reversed. E.g. [4,4,2] -> encode [4,2,2,1] "
            "-> reversed [1,2,2,4]."
        ),
        # Stage 1: run-length encode into a flat [v,k,v,k,...] list. Stage 2: reverse it.
        reference=(
            "def solve(xs):\n"
            "  enc = []\n"
            "  i = 0\n"
            "  n = len(xs)\n"
            "  while i < n:\n"
            "    v = xs[i]\n"
            "    k = 0\n"
            "    while i < n and xs[i] == v:\n"
            "      k += 1\n"
            "      i += 1\n"
            "    enc.append(v)\n"
            "    enc.append(k)\n"
            "  return enc[::-1]"
        ),
        gen=lambda rng: (_rand_run_list(rng),),
    ),
    _Family(
        key="l8_top2_digitcount",
        level=8,
        signature="solve(xs)",
        description=(
            "Add together the two largest DISTINCT values of xs (ignore "
            "duplicates), then return how many decimal digits that sum has. xs "
            "has at least two distinct positive values."
        ),
        # Stage 1: dedupe. Stage 2: sort + add the top two. Stage 3: count digits.
        reference=(
            "def solve(xs):\n"
            "  uniq = []\n"
            "  for x in xs:\n"
            "    if x not in uniq:\n"
            "      uniq.append(x)\n"
            "  uniq = sorted(uniq)\n"
            "  total = uniq[-1] + uniq[-2]\n"
            "  if total < 0:\n"
            "    total = -total\n"
            "  if total == 0:\n"
            "    return 1\n"
            "  c = 0\n"
            "  while total > 0:\n"
            "    c += 1\n"
            "    total //= 10\n"
            "  return c"
        ),
        gen=lambda rng: (_rand_dedup_list(rng, 1, 60),),
    ),
    _Family(
        key="l8_eval_expr",
        level=8,
        signature="solve(s)",
        description=(
            "The string s is an arithmetic expression of the form 'a+b', 'a-b' "
            "or 'a*b' where a and b are non-negative integers. Parse it and "
            "return the integer result."
        ),
        # Stage 1: find the operator + parse the two integer operands. Stage 2:
        # dispatch on the operator and compute.
        reference=(
            "def solve(s):\n"
            "  for op in '+-*':\n"
            "    i = s.find(op)\n"
            "    if i > 0:\n"
            "      a = int(s[:i])\n"
            "      b = int(s[i+1:])\n"
            "      if op == '+':\n"
            "        return a + b\n"
            "      if op == '-':\n"
            "        return a - b\n"
            "      return a * b\n"
            "  return int(s)"
        ),
        gen=lambda rng: (_rand_arith_expr(rng),),
    ),
    _Family(
        key="l8_collatz_max",
        level=8,
        signature="solve(n)",
        description=(
            "Run the Collatz process from the positive integer n (each step: "
            "n -> n//2 if even, else n -> 3*n+1, stopping at 1) and return the "
            "largest value reached anywhere along the way, including n itself."
        ),
        # Stage 1: generate the Collatz trajectory. Stage 2: track its running max.
        reference=(
            "def solve(n):\n"
            "  best = n\n"
            "  while n != 1:\n"
            "    if n % 2 == 0:\n"
            "      n //= 2\n"
            "    else:\n"
            "      n = 3 * n + 1\n"
            "    if n > best:\n"
            "      best = n\n"
            "  return best"
        ),
        gen=lambda rng: (rng.choice([1, 2, rng.randint(3, 40)]),),
    ),
    # --- Level 9: hardest multi-stage -------------------------------------
    _Family(
        key="l9_prime_digitsum",
        level=9,
        signature="solve(n)",
        description=(
            "For every prime p with p <= n, compute the sum of p's decimal "
            "digits, and return the total over all such primes (so for n=13 the "
            "primes are 2,3,5,7,11,13 with digit sums 2,3,5,7,2,4 totalling 23). "
            "Return 0 when n < 2."
        ),
        # Stage 1: enumerate primes <= n. Stage 2: per-prime digit sum. Stage 3:
        # accumulate the grand total.
        reference=(
            "def solve(n):\n"
            "  total = 0\n"
            "  m = 2\n"
            "  while m <= n:\n"
            "    is_p = True\n"
            "    d = 2\n"
            "    while d * d <= m:\n"
            "      if m % d == 0:\n"
            "        is_p = False\n"
            "        break\n"
            "      d += 1\n"
            "    if is_p:\n"
            "      t = m\n"
            "      while t > 0:\n"
            "        total += t % 10\n"
            "        t //= 10\n"
            "    m += 1\n"
            "  return total"
        ),
        # Bias toward n>=10 (and seed 2,3) so trivial constants fail.
        gen=lambda rng: (
            rng.choice([2, 3, rng.randint(10, 60), rng.randint(10, 60), rng.randint(10, 60)]),
        ),
    ),
    _Family(
        key="l9_top_scorer",
        level=9,
        signature="solve(s)",
        description=(
            "The string s holds space-separated 'name:score' entries (score is a "
            "non-negative integer), e.g. 'amy:7 ben:3'. Parse them and return the "
            "name with the highest score (the maximum score is unique)."
        ),
        # Stage 1: split into entries and parse each name:score pair. Stage 2:
        # argmax over the parsed scores.
        reference=(
            "def solve(s):\n"
            "  pairs = s.split(' ')\n"
            "  best = ''\n"
            "  best_score = -1\n"
            "  for p in pairs:\n"
            "    i = p.find(':')\n"
            "    name = p[:i]\n"
            "    score = int(p[i+1:])\n"
            "    if score > best_score:\n"
            "      best_score = score\n"
            "      best = name\n"
            "  return best"
        ),
        gen=lambda rng: (_rand_score_sentence(rng),),
    ),
    _Family(
        key="l9_max_window3",
        level=9,
        signature="solve(xs)",
        description=(
            "Consider every window of 3 consecutive elements of xs, sum each "
            "window, and return the largest window sum. xs has at least three "
            "elements."
        ),
        # Stage 1: slide a width-3 window summing each. Stage 2: take the max sum.
        reference=(
            "def solve(xs):\n"
            "  best = xs[0] + xs[1] + xs[2]\n"
            "  i = 0\n"
            "  n = len(xs)\n"
            "  while i + 3 <= n:\n"
            "    s = xs[i] + xs[i+1] + xs[i+2]\n"
            "    if s > best:\n"
            "      best = s\n"
            "    i += 1\n"
            "  return best"
        ),
        gen=lambda rng: (_rand_int_list(rng, 3, 8, -9, 9, allow_dups=True),),
    ),
]

_FAMILIES_BY_LEVEL: dict[int, list[_Family]] = {}
for _fam in _FAMILIES:
  _FAMILIES_BY_LEVEL.setdefault(_fam.level, []).append(_fam)


# ---------------------------------------------------------------------------
# Random input generators (shared by the family ``gen`` lambdas).
# ---------------------------------------------------------------------------


def _rand_int_list(
    rng: random.Random,
    lo_len: int,
    hi_len: int,
    lo_val: int,
    hi_val: int,
    *,
    allow_dups: bool,
) -> list[int]:
  """A random int list of length in ``[lo_len, hi_len]``.

  When ``allow_dups`` we occasionally force a duplicate so families that care
  about ties / repeated maxima (max_of_list, second_largest) actually exercise
  that path across the test set.
  """
  n = rng.randint(lo_len, hi_len)
  xs = [rng.randint(lo_val, hi_val) for _ in range(n)]
  if allow_dups and n >= 2 and rng.random() < 0.5:
    # Duplicate a random element into another slot.
    src = rng.randrange(n)
    dst = rng.randrange(n)
    xs[dst] = xs[src]
  return xs


def _rand_word(rng: random.Random, lo_len: int, hi_len: int) -> str:
  """A random lowercase word of length in ``[lo_len, hi_len]`` (may be empty)."""
  alpha = "abcdefghijklmnopqrstuvwxyz"
  n = rng.randint(lo_len, hi_len)
  return "".join(rng.choice(alpha) for _ in range(n))


def _rand_runs(rng: random.Random) -> str:
  """A lowercase string built from a few character runs (for run-length encode).

  Occasionally empty so the ``'' -> ''`` edge case is exercised.
  """
  if rng.random() < 0.15:
    return ""
  alpha = "abcde"
  out = []
  for _ in range(rng.randint(1, 4)):
    ch = rng.choice(alpha)
    out.append(ch * rng.randint(1, 4))
  return "".join(out)


def _rand_sentence(rng: random.Random) -> str:
  """A space-separated sentence drawn from a tiny vocabulary (forces repeats)."""
  vocab = ["cat", "dog", "bird", "fish", "fox", "owl"]
  n = rng.randint(2, 7)
  return " ".join(rng.choice(vocab) for _ in range(n))


def _rand_palindrome_candidate(rng: random.Random) -> str:
  """A sentence that is, with ~50% probability, a (space-insensitive) palindrome."""
  alpha = "abcde"
  core = "".join(rng.choice(alpha) for _ in range(rng.randint(1, 5)))
  if rng.random() < 0.5:
    # Make a genuine space-insensitive palindrome, then sprinkle in a space.
    full = core + core[::-1]
  else:
    full = core + "".join(rng.choice(alpha) for _ in range(rng.randint(1, 4)))
  # Optionally insert a single space so the "remove spaces" step matters.
  if len(full) >= 2 and rng.random() < 0.6:
    cut = rng.randint(1, len(full) - 1)
    full = full[:cut] + " " + full[cut:]
  return full


# --- generators for the multi-stage levels 7-9 -------------------------------
#
# These deliberately bias away from the trivial small inputs (n<2, all-equal
# lists) so the constant ``return 0`` / first-arg programs fail the hidden tests,
# while still occasionally seeding the edge cases the references must handle.


def _rand_dedup_list(rng: random.Random, lo_val: int, hi_val: int) -> list[int]:
  """An int list of length 3-7 with >=2 distinct values and seeded duplicates.

  Used by the "second-largest *distinct*" / "two-largest *distinct*" families,
  where the dedupe stage only matters when the maximum is repeated -- so we force
  a duplicate into the list ~60% of the time.
  """
  while True:
    n = rng.randint(3, 7)
    xs = [rng.randint(lo_val, hi_val) for _ in range(n)]
    if rng.random() < 0.6:
      src = rng.randrange(n)
      dst = rng.randrange(n)
      xs[dst] = xs[src]
    if len(set(xs)) >= 2:
      return xs


def _rand_vowel_sentence(rng: random.Random) -> str:
  """A space-separated sentence of random lowercase words (2-6 letters each)."""
  n = rng.randint(2, 5)
  return " ".join(_rand_word(rng, 2, 6) for _ in range(n))


def _rand_run_list(rng: random.Random) -> list[int]:
  """An int list built from a few runs of repeated values (for run-length encode)."""
  out: list[int] = []
  for _ in range(rng.randint(1, 4)):
    v = rng.randint(0, 4)
    out.extend([v] * rng.randint(1, 4))
  return out


def _rand_arith_expr(rng: random.Random) -> str:
  """A string ``"a<op>b"`` with a single binary operator (op in ``+ - *``)."""
  a = rng.randint(1, 99)
  b = rng.randint(1, 99)
  op = rng.choice(["+", "-", "*"])
  return str(a) + op + str(b)


def _rand_score_sentence(rng: random.Random) -> str:
  """A ``"name:score name:score ..."`` string with a *unique* maximum score.

  The unique max avoids tie ambiguity so the family has one well-defined answer.
  """
  n = rng.randint(2, 5)
  names: list[str] = []
  used: set = set()
  while len(names) < n:
    w = _rand_word(rng, 2, 4)
    if w not in used:
      used.add(w)
      names.append(w)
  while True:
    scores = [rng.randint(0, 30) for _ in range(n)]
    top = max(scores)
    if scores.count(top) == 1:
      break
  return " ".join(names[i] + ":" + str(scores[i]) for i in range(n))


# ---------------------------------------------------------------------------
# Building test cases (expected output computed by running the reference under
# micropython, so the gold formatting matches the engine the agent runs in).
# ---------------------------------------------------------------------------


def _call_source(program: str, args: tuple) -> str:
  """The micropython source that calls ``solve`` on ``args`` and prints the result.

  micropython has no ``*args`` call form, so we expand the tuple into a literal
  positional argument list: ``solve(3, 4)`` / ``solve([1, 2, 3])`` /
  ``solve('hi', 2)``. ``repr`` round-trips through the interpreter's parser for
  every value our generators produce (ints, lists of ints, strings).
  """
  inner = ", ".join(repr(a) for a in args)
  return program + "\nprint(solve(" + inner + "))"


def _expected_output(reference: str, args: tuple) -> str:
  """The gold stdout for ``args``, computed by running the reference oracle."""
  result = micropython.run(_call_source(reference, args), max_steps=_GRADE_MAX_STEPS)
  if not result.ok:
    # A reference that errors on a generated instance is a bug in the family; the
    # self-check asserts this never happens.
    raise RuntimeError(
        f"reference errored on args={args!r}: {result.error}"
    )
  return result.stdout


def _make_tests(
    family: _Family, rng: random.Random, n_tests: int
) -> list[tuple[tuple, str]]:
  """Generate ``n_tests`` distinct ``(args, expected)`` cases for one instance.

  We dedupe on the arg tuple (so the public/hidden split shows genuinely
  different inputs) and over-sample to fill the quota even when the generator's
  support is small.
  """
  tests: list[tuple[tuple, str]] = []
  seen: set = set()
  attempts = 0
  while len(tests) < n_tests and attempts < n_tests * 50:
    attempts += 1
    args = family.gen(rng)
    key = repr(args)
    if key in seen:
      continue
    seen.add(key)
    tests.append((args, _expected_output(family.reference, args)))
  return tests


# ---------------------------------------------------------------------------
# Sampling / loading problems.
# ---------------------------------------------------------------------------


def _instance_id(family: _Family, rng_seed_token: int) -> str:
  """A stable per-(family, instance) id slug."""
  return f"{family.key}#{rng_seed_token & 0xFFFF:04x}"


def _build_problem(family: _Family, rng: random.Random) -> Problem:
  """Build one :class:`Problem` for ``family`` using ``rng`` for its instance."""
  n_tests = max(_MIN_TESTS, 7)
  tests = _make_tests(family, rng, n_tests)
  public = tests[:_N_PUBLIC]
  hidden = tests[_N_PUBLIC:]
  # Derive a stable token from the tests so the id is reproducible for a given
  # generated instance (the same draw yields the same id).
  token = hash(repr([a for a, _ in tests])) & 0x7FFFFFFF
  problem = Problem(
      id=_instance_id(family, token),
      level=family.level,
      family=family.key,
      prompt="",  # filled below by format_problem_prompt
      public_tests=public,
      hidden_tests=hidden,
  )
  problem.prompt = format_problem_prompt(problem, family)
  return problem


def sample_problem(rng: random.Random, level: int) -> Problem:
  """Sample a random problem instance at ``level`` (1..NUM_LEVELS)."""
  if level not in _FAMILIES_BY_LEVEL:
    raise ValueError(f"no families for level {level}")
  family = rng.choice(_FAMILIES_BY_LEVEL[level])
  return _build_problem(family, rng)


def load_eval_problems(level: int, n: int, seed: int) -> list[Problem]:
  """Return ``n`` fixed held-out problem instances at ``level``.

  Deterministic for a fixed ``(level, n, seed)``: it cycles through the level's
  families round-robin so the held-out set is balanced across families, and
  seeds each instance from a distinct, reproducible sub-seed.
  """
  if level not in _FAMILIES_BY_LEVEL:
    raise ValueError(f"no families for level {level}")
  families = _FAMILIES_BY_LEVEL[level]
  problems: list[Problem] = []
  for i in range(n):
    family = families[i % len(families)]
    rng = random.Random((seed * 1_000_003) ^ (level * 7919) ^ (i * 104_729))
    problems.append(_build_problem(family, rng))
  return problems


# ---------------------------------------------------------------------------
# Prompt formatting.
# ---------------------------------------------------------------------------


def _example_line(args: tuple, expected: str) -> str:
  """Render one public example as ``solve(args) -> output`` for the prompt."""
  call = "solve(" + ", ".join(repr(a) for a in args) + ")"
  return f"  {call} -> {expected.rstrip(chr(10))!r}"


def format_problem_prompt(problem: Problem, family: "_Family | None" = None) -> str:
  """The human task text: signature + behaviour + public examples + END instruction.

  Mentions ``solve`` and ends by instructing the model to write ``def solve(...)``
  followed by a line containing only ``END`` (the program-extraction convention
  of :mod:`coding_agent_env`). The public examples are shown as
  ``solve(args) -> printed-output`` lines; hidden tests are never mentioned.
  """
  if family is None:
    family = _family_for(problem.family)
  lines = [
      f"Write a function {family.signature}.",
      family.description,
  ]
  if problem.public_tests:
    lines.append("Examples (the printed output of solve must equal the value shown):")
    for args, expected in problem.public_tests:
      lines.append(_example_line(args, expected))
  lines.append(
      "Define def solve(...) and then write a line containing only END."
  )
  return "\n".join(lines)


def _family_for(key: str) -> _Family:
  for fam in _FAMILIES:
    if fam.key == key:
      return fam
  raise KeyError(key)


def reference_for(problem: Problem) -> str:
  """The family's correct ``def solve(...)`` source (for SFT warm-up transcripts).

  Verified in the self-check to pass all of the problem's tests in micropython.
  """
  return _family_for(problem.family).reference


# ---------------------------------------------------------------------------
# Grading.
# ---------------------------------------------------------------------------


def _has_solve(program: str) -> bool:
  """True if the program defines ``solve`` (cheap textual check, like grade_program)."""
  return "def solve" in (program or "")


def grade_problem(
    program: str, problem: Problem, *, max_steps: int = _GRADE_MAX_STEPS
) -> dict:
  """Grade ``program`` against all (public + hidden) tests of ``problem``.

  Returns the dense components:

    * ``has_code``: 1.0 if the program defines ``solve``, else 0.0 (and every
      other component is 0.0).
    * ``ran_ok``: fraction of tests whose execution did NOT error.
    * ``frac_passed``: fraction of tests whose stdout exactly matched the gold.
    * ``n_tests``: the total number of tests graded.
    * ``exact``: 1.0 iff ``frac_passed == 1.0``, else 0.0.

  For each test ``(args, expected)`` the program is run as
  ``program + "\\nprint(solve(<args>))"`` through micropython (the same engine
  that produced ``expected``), so a correct program reproduces the gold byte for
  byte.
  """
  tests = problem.public_tests + problem.hidden_tests
  n = len(tests)
  if not _has_solve(program):
    return {
        "has_code": 0.0,
        "ran_ok": 0.0,
        "frac_passed": 0.0,
        "n_tests": n,
        "exact": 0.0,
    }
  n_ran_ok = 0
  n_passed = 0
  for args, expected in tests:
    result = micropython.run(_call_source(program, args), max_steps=max_steps)
    if result.ok:
      n_ran_ok += 1
      if result.stdout == expected:
        n_passed += 1
  frac_passed = (n_passed / n) if n else 0.0
  return {
      "has_code": 1.0,
      "ran_ok": (n_ran_ok / n) if n else 0.0,
      "frac_passed": frac_passed,
      "n_tests": n,
      "exact": 1.0 if frac_passed >= 1.0 else 0.0,
  }


# Dense reward weights (CURRICULUM_DESIGN.md §1): a tiny credit for emitting code
# at all, a moderate credit for code that runs without error, and the bulk for
# the fraction of tests passed -- so a Dr.GRPO group has continuous reward
# variance even before any sample is fully correct.
_W_HAS_CODE = 0.10
_W_RAN_OK = 0.20
_W_FRAC = 0.70


def problem_reward(components: dict) -> float:
  """The dense scalar reward from :func:`grade_problem`'s components.

  ``0.10*has_code + 0.20*ran_ok + 0.70*frac_passed``. A full pass scores exactly
  1.0; an all-zero grade scores 0.0; a partial pass lands strictly in between.
  """
  return (
      _W_HAS_CODE * components.get("has_code", 0.0)
      + _W_RAN_OK * components.get("ran_ok", 0.0)
      + _W_FRAC * components.get("frac_passed", 0.0)
  )


def format_test_feedback(
    program: str, problem: Problem, *, max_steps: int = _GRADE_MAX_STEPS
) -> str:
  """The multi-turn ``Tool result:`` body for ``program`` on the PUBLIC tests.

  Shows, for each public test, ``solve(args) -> got vs expected`` and a
  pass/fail count, truncated to ~``_MAX_FEEDBACK_CHARS`` chars. Hidden tests are
  never revealed (neither their inputs nor their outputs), matching the design's
  "the public tests + a failing-test summary are the feedback" contract while
  keeping the held-out set unseen.
  """
  if not _has_solve(program):
    return "(no def solve found)"
  lines: list[str] = []
  passed = 0
  for args, expected in problem.public_tests:
    result = micropython.run(_call_source(program, args), max_steps=max_steps)
    call = "solve(" + ", ".join(repr(a) for a in args) + ")"
    if not result.ok:
      got = f"error: {result.error}"
    else:
      got = repr(result.stdout.rstrip(chr(10)))
      if result.stdout == expected:
        passed += 1
    lines.append(f"{call} -> {got} (expected {expected.rstrip(chr(10))!r})")
  header = f"public tests passed {passed}/{len(problem.public_tests)}"
  body = "\n".join([header] + lines)
  if len(body) > _MAX_FEEDBACK_CHARS:
    body = body[: _MAX_FEEDBACK_CHARS - 14] + "...(truncated)"
  return body

