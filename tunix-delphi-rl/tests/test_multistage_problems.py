# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for the multi-stage (compositional) families: levels 7-9.

These levels were added to give a 2B-class model a HARDER problem surface than
the level 1-6 single-function families: each ``solve`` computes an intermediate
result and then reduces / transforms it, so a buggy first attempt is often
*partially* right (the pass@k > pass@1 gap RL can sharpen). The checks here are
the family-specific complement to ``test_coding_problems.py``:

  * every level-7/8/9 reference scores ``exact == 1.0`` on several sampled
    instances (its program is micropython-valid and correct), and
  * a deliberately-wrong stub for each family scores ``frac_passed < 1.0`` on at
    least one sampled instance -- proving the per-instance tests genuinely
    discriminate a partial solution (the partial-credit gradient).
"""

from __future__ import annotations

import random

import problems.coding_problems as cp
from problems.coding_problems import (
    NUM_LEVELS,
    grade_problem,
    problem_reward,
)

# The new multi-stage levels and the number of families we expect at each.
MULTISTAGE_LEVELS = (7, 8, 9)

# For each new family: a deliberately-wrong ``solve`` that gets ONE stage wrong,
# so it passes some instances/tests but not all (frac_passed < 1.0 somewhere).
# The keys must exactly cover every family at levels 7-9.
_WRONG_STUBS: dict[str, str] = {
    # Returns n instead of the sum of primes <= n.
    "l7_sum_primes": "def solve(a=0, b=0, c=0):\n  return a",
    # Forgets to dedupe: second-from-top of the RAW sorted list (wrong when the
    # maximum is duplicated).
    "l7_second_distinct_sq": (
        "def solve(xs):\n"
        "  s = sorted(xs)\n"
        "  v = s[-2]\n"
        "  return v * v"
    ),
    # Returns the LONGEST word, not the most-vowel word.
    "l7_most_vowels_word": (
        "def solve(s):\n"
        "  words = s.split(' ')\n"
        "  best = words[0]\n"
        "  for w in words:\n"
        "    if len(w) > len(best):\n"
        "      best = w\n"
        "  return best"
    ),
    # Digit sum of n, not of n!.
    "l7_fact_digit_sum": (
        "def solve(n):\n"
        "  total = 0\n"
        "  while n > 0:\n"
        "    total += n % 10\n"
        "    n //= 10\n"
        "  return total"
    ),
    # Builds the encoding but forgets the final reverse stage.
    "l8_rle_reversed": (
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
        "  return enc"
    ),
    # Digit count of the single max, skipping the "add the top two distinct" stage.
    "l8_top2_digitcount": (
        "def solve(xs):\n"
        "  m = max(xs)\n"
        "  c = 0\n"
        "  while m > 0:\n"
        "    c += 1\n"
        "    m //= 10\n"
        "  return c"
    ),
    # Only handles '+'; mishandles '-' / '*' expressions.
    "l8_eval_expr": (
        "def solve(s):\n"
        "  i = s.find('+')\n"
        "  a = int(s[:i])\n"
        "  b = int(s[i+1:])\n"
        "  return a + b"
    ),
    # Near-miss: seeds the running max with 1 instead of n, so it forgets that
    # the starting value n can itself be the trajectory maximum (wrong exactly
    # when n is the largest value reached, e.g. n a power of two).
    "l8_collatz_max": (
        "def solve(n):\n"
        "  best = 1\n"
        "  while n != 1:\n"
        "    if n % 2 == 0:\n"
        "      n //= 2\n"
        "    else:\n"
        "      n = 3 * n + 1\n"
        "    if n > best:\n"
        "      best = n\n"
        "  return best"
    ),
    # Sums the primes themselves instead of their digit sums.
    "l9_prime_digitsum": (
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
    # Argmax over name length, ignoring the parsed score.
    "l9_top_scorer": (
        "def solve(s):\n"
        "  pairs = s.split(' ')\n"
        "  best = ''\n"
        "  for p in pairs:\n"
        "    i = p.find(':')\n"
        "    name = p[:i]\n"
        "    if len(name) > len(best):\n"
        "      best = name\n"
        "  return best"
    ),
    # Returns the FIRST window's sum, skipping the slide-and-max stage.
    "l9_max_window3": "def solve(xs):\n  return xs[0] + xs[1] + xs[2]",
}


def _families_for(level: int):
  return cp._FAMILIES_BY_LEVEL[level]


def test_num_levels_extended_to_nine():
  # The multi-stage levels were appended without disturbing the 1-6 contract.
  assert NUM_LEVELS == 9
  for level in MULTISTAGE_LEVELS:
    fams = _families_for(level)
    assert 3 <= len(fams) <= 4, (level, len(fams))


def test_stub_table_covers_every_multistage_family():
  # Every level-7/8/9 family has exactly one wrong-stub entry (no orphans).
  keys = set()
  for level in MULTISTAGE_LEVELS:
    for fam in _families_for(level):
      keys.add(fam.key)
  assert keys == set(_WRONG_STUBS), (keys ^ set(_WRONG_STUBS))


def test_multistage_references_are_exact():
  # Each multi-stage reference scores exact==1.0 on several sampled instances,
  # i.e. it is micropython-valid AND correct on every one of its own tests.
  for level in MULTISTAGE_LEVELS:
    for fam in _families_for(level):
      for seed in range(5):
        rng = random.Random(1000 * level + 31 * seed + 1)
        problem = cp._build_problem(fam, rng)
        assert problem.level == level
        assert len(problem.public_tests) <= cp._N_PUBLIC
        assert len(problem.hidden_tests) >= cp._MIN_HIDDEN, (fam.key, problem)
        grade = grade_problem(fam.reference, problem)
        assert grade["ran_ok"] == 1.0, (fam.key, seed, grade)
        assert grade["frac_passed"] == 1.0, (fam.key, seed, grade)
        assert grade["exact"] == 1.0, (fam.key, seed, grade)
        assert abs(problem_reward(grade) - 1.0) < 1e-9, (fam.key, seed, grade)


def test_multistage_wrong_stubs_score_partial():
  # For each family, the deliberately-wrong stub scores frac_passed < 1.0 on at
  # least one sampled instance (the per-instance tests discriminate a partial
  # solution), while still registering has_code == 1.0.
  for level in MULTISTAGE_LEVELS:
    for fam in _families_for(level):
      stub = _WRONG_STUBS[fam.key]
      saw_discriminated = False
      for seed in range(8):
        rng = random.Random(7000 * level + 53 * seed + 3)
        problem = cp._build_problem(fam, rng)
        grade = grade_problem(stub, problem)
        assert grade["has_code"] == 1.0, (fam.key, seed, grade)
        if grade["frac_passed"] < 1.0:
          saw_discriminated = True
      assert saw_discriminated, (
          f"{fam.key}: wrong stub passed ALL tests on every sampled instance "
          "(the tests do not discriminate this bug)"
      )


def test_at_least_one_instance_has_strictly_partial_reward():
  # Across the multi-stage families, the wrong stubs produce a strictly-interior
  # reward on at least one instance (0 < reward < 1) -- the dense gradient these
  # harder levels are meant to expose. We assert it holds for every family.
  for level in MULTISTAGE_LEVELS:
    for fam in _families_for(level):
      stub = _WRONG_STUBS[fam.key]
      saw_strict_partial = False
      for seed in range(12):
        rng = random.Random(20000 * level + 17 * seed + 5)
        problem = cp._build_problem(fam, rng)
        grade = grade_problem(stub, problem)
        reward = problem_reward(grade)
        if 0.0 < grade["frac_passed"] < 1.0:
          assert 0.0 < reward < 1.0, (fam.key, seed, grade, reward)
          saw_strict_partial = True
      assert saw_strict_partial, (
          f"{fam.key}: never observed a strictly-partial frac_passed across "
          "12 instances (stub is either always right or always all-wrong)"
      )
